import numpy as np
import gzip
import os

def sampling_hic(hic_matrix, sampling_ratio, fix_seed=False):
    """sampling dense hic matrix"""
    m = np.matrix(hic_matrix)
    all_sum = m.sum(dtype='float')
    idx_prob = np.divide(m, all_sum, out=np.zeros_like(m), where=all_sum != 0)
    idx_prob = np.asarray(idx_prob.reshape(
        (idx_prob.shape[0]*idx_prob.shape[1],)))
    idx_prob = np.squeeze(idx_prob)
    sample_number_counts = int(all_sum/(2*sampling_ratio))
    id_range = np.arange(m.shape[0]*m.shape[1])
    if fix_seed:
        np.random.seed(0)
    id_x = np.random.choice(
        id_range, size=sample_number_counts, replace=True, p=idx_prob)
    sample_m = np.zeros_like(m)
    for i in np.arange(sample_number_counts):
        x = int(id_x[i]/m.shape[0])
        y = int(id_x[i] % m.shape[0])
        sample_m[x, y] += 1.0
    sample_m = np.transpose(sample_m) + sample_m
    return np.asarray(sample_m)


def divide_pieces_hic(hic_matrix, block_size=128, max_distance=None, save_file=False, pathfile=None):
    M = hic_matrix

    IMG_HEIGHT, IMG_WIDTH = int(block_size), int(block_size)
    print('Height: ', IMG_HEIGHT, 'Weight: ', IMG_WIDTH)
    M_h, M_w = M.shape
    block_height = int(IMG_HEIGHT/2)
    block_width = int(IMG_WIDTH/2)
    M_d0 = np.split(M, np.arange(block_height, M_h, block_height), axis=0)
    M_d1 = list(map(lambda x: np.split(x, np.arange(
        block_width, M_w, block_width), axis=1), M_d0))
    hic_half_h = np.array(M_d1)
    if M_h%block_height!=0 or M_w%block_width!=0:
        hic_half_h = hic_half_h[0:-1, 0:-1]
    print('shape of blocks: ', hic_half_h.shape)

    hic_m = list()
    hic_index = dict()
    hic_index_rev = dict()
    count = 0
    for dis in np.arange(1, hic_half_h.shape[0]):
        for i in np.arange(0, hic_half_h.shape[1]-dis):
            if (max_distance is not None) and (dis>max_distance):
                continue
            hic_m.append(np.block([[hic_half_h[i, i], hic_half_h[i, i+dis]],
                                   [hic_half_h[i+dis, i], hic_half_h[i+dis, i+dis]]]))
            hic_index[count] = (i, i+dis)
            hic_index_rev[(i, i+dis)] = count
            count = count + 1
    print('# of hic pieces: ', len(hic_m))

    if save_file:
        from numpy import savez_compressed, savez
        if pathfile is None:
            pathfile = './datasets_hic'
        savez_compressed(pathfile+'.npz', hic=hic_m,
                         index_1D_2D=hic_index, index_2D_1D=hic_index_rev)

    return hic_m, hic_index, hic_index_rev


def merge_hic(hic_lists, index_1D_2D):
    hic_m = np.asarray(hic_lists)
    lensize, Height, Width = hic_m.shape
    lenindex = len(index_1D_2D)
    print('lenindex: ', lenindex)
    if lenindex != lensize:
        raise 'ERROR dimension must equal. length of hic list: ' + lensize + \
            'is not equal to length of index_1D_2D: ' + len(index_1D_2D)

    if 2*lenindex != int(np.sqrt(2*lenindex))*(int(np.sqrt(2*lenindex))+1):
        raise 'ERROR: not square'

    n = int(np.sqrt(2*lenindex)+1)
    Height_hf = int(Height/2)
    Width_hf = int(Width/2)
    matrix = np.zeros(shape=(n*Height_hf, n*Width_hf))
    for i in np.arange(lenindex):
        h, w = index_1D_2D[i]
        x = h*Height_hf
        y = w*Width_hf
        matrix[x:x+Height_hf, y:y+Width_hf] += hic_m[i, 0:Height_hf, 0+Width_hf:Width]

        matrix[x:x+Height_hf, x:x+Height_hf] += hic_m[i, 0:Height_hf, 0:Width_hf]/(2.0*(n-1))
        matrix[y:y+Width_hf, y:y+Width_hf] += hic_m[i, 0+Height_hf:Height, 0+Width_hf:Width]/(2.0*(n-1))

    matrix = matrix + np.transpose(matrix)
    return matrix

def filter_diag_boundary(hic, diag_k=0, boundary_k=None):
    if boundary_k is None:
        boundary_k = hic.shape[0]-1
    filter_m = np.tri(N = hic.shape[0], k=boundary_k)
    filter_m = np.triu(filter_m, k=diag_k)
    filter_m = filer_m + np.transpose(filter_m)
    return np.multiply(hic, filter_m)

def dense2tag(matrix):
    """converting a square matrix (dense) to coo-based tag matrix"""
    Height, Width = matrix.shape
    tag_mat = list()
    for i in np.arange(Height):
        for j in np.arange(i+1, Width):
            if float(matrix[i, j]) > 1.0e-20:
                tag_mat.append([int(i), int(j), float(matrix[i, j])])
    tag_mat = np.asarray(tag_mat, dtype=np.float)
    return tag_mat


def tag2dense(tag_mat, mat_length):
    """converting a tag matrix to square matrix (dense)"""
    Height, Width = int(mat_length), int(mat_length)
    matrix = np.zeros(shape=(Height, Width))

    for i in np.arange(len(tag_mat)):
        x, y, c = tag_mat[i]
        matrix[int(x), int(y)] = float(c)

    return tag_mat


def format_contact(matrix, coordinate=(0, 1), resolution=10000, chrm='1', save_file=True, filename=None):
    """chr1 bin1 chr2 bin2 value"""
    n = len(matrix)
    nhf = int(len(matrix)/2)
    contact = list()
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            value = float(matrix[i, j])
            if value <= 1.0e-10:
                continue
            chr1 = chrm
            chr2 = chrm
            bin1 = (i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution
            bin2 = (j - int(j/nhf)*nhf + coordinate[int(j/nhf)]*nhf)*resolution
            entry = [chr1, str(bin1), chr2, str(bin2), str(value)]
            contact.append('\t'.join(entry))
    contact_txt = "\n".join(contact)
    #contact_txt = format(contact_txt, 'b')
    if save_file:
        if filename is None:
            filename = './demo_contact.gz'
        output = gzip.open(filename, 'w+')
        try:
            output.write(contact_txt.encode())
        finally:
            output.close()
    return contact


def format_bin(matrix, coordinate=(0, 1), resolution=10000, chrm='1', save_file=True, filename=None):
    """chr start end name"""
    n = len(matrix)
    nhf = int(len(matrix)/2)
    bins = list()

    for i in np.arange(n):
        chr1 = chrm
        start = int((i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution)
        end = int(start + resolution)
        entry = [chr1, str(start), str(end), str(start)]
        bins.append('\t'.join(entry))
    if save_file:
        if filename is None:
            filename = './demo.bed.gz'
        file = gzip.open(filename,"w+")
        for l in bins:
            line = l + '\n'
            file.write(line.encode())
        file.close()
    return bins


def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy

