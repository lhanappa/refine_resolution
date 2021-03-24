# The authors use a random down-sampling procedure to generate the low-resolution Hi-C matrices.
# How well do these simulated low-resolution Hi-C matrices mimic experimentally-determined low-resolution Hi-C matrices?

# https://data.4dnucleome.org/experiment-set-replicates/4DNES3RHDBBR/#processed-files
# 4 replicates: 4DNEXNMDXSP1, 4DNEXYNM1II5, 4DNEXT3VJDNU, 4DNEXWGWFK1T
# 4 files:      4DNFIA32ODXZ, 4DNFIAK2SX23, 4DNFIMEIXLSE, 4DNFIH6KL2OT
# combined: 4DNES3RHDBBR (4DNFI9PIEPQA.mcool)
import cooler
import sys, os
import numpy as np
import pandas
import gzip

# bash hg38
# >> pairix 4DNFIA32ODXZ.pairs.gz
# >> cooler cload pairix hg38.chrom.sizes:10000 4DNFIA32ODXZ.pairs.gz hic.cool


replication = {'rep1':'4DNFIA32ODXZ', 
            'rep2': '4DNFIAK2SX23', 
            'rep3': '4DNFIMEIXLSE', 
            'rep4': '4DNFIH6KL2OT',
            'multiple': '4DNFI9PIEPQA'}

def sampling_hic(hic_matrix, sampling_ratio, fix_seed=False):
    """sampling dense hic matrix"""
    m = np.matrix(hic_matrix, dtype='float')
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

def format_contact(matrix, resolution=10000, chrm='1', save_file=True, filename=None):
    """chr1 bin1 chr2 bin2 value"""
    n = len(matrix)
    #nhf = np.floor(n/2)
    contact = list()
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            value = float(matrix[i, j])
            if value <= 1.0e-10:
                continue
            chr1 = chrm
            chr2 = chrm
            # print('i: {}, j: {}, nhf: {}, int(i/nhf): {}, int(j/nhf): {}'.format(i, j, nhf, int(i/nhf), int(j/nhf)))
            # bin1 = (i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution
            # bin2 = (j - int(j/nhf)*nhf + coordinate[int(j/nhf)]*nhf)*resolution
            bin1 = i*resolution
            bin2 = j*resolution
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

def format_bin(matrix, resolution=10000, chrm='1', save_file=True, filename=None):
    """chr start end name"""
    n = len(matrix)
    # nhf = int(len(matrix)/2)
    bins = list()

    for i in np.arange(n):
        chr1 = 'chr' + chrm
        start = int(i*resolution)
        # start = int((i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution)
        end = int(start + resolution)
        entry = [chr1, str(start), str(end), str(start)]
        bins.append('\t'.join(entry))
    if save_file:
        if filename is None:
            filename = './demo.bed.gz'
        file = gzip.open(filename, "w+")
        for l in bins:
            line = l + '\n'
            file.write(line.encode())
        file.close()
    return bins

def load_hic_pixel(path, name, chrom):
    hic = cooler.Cooler(os.path.join(path, name))
    return hic.matrix(balance=False, as_pixels=False).fetch(chrom)

def split_chrom(path, name, ftype, chrom, resolution):
    hic = load_hic_pixel(path, name, chrom)
    if ftype == 'multiple': 
        sampling_ratio = 4
        hic = sampling_hic(hic, sampling_ratio, fix_seed=True)
    output = os.path.join('.', 'data')
    os.makedirs(output, exist_ok=True)
    filename = os.path.join(output, '{}_{}_bed.gz'.format(ftype, chrom))
    format_bin(hic, resolution=resolution, chrm=chrom, save_file=True, filename=filename)
    filename = os.path.join(output, '{}_{}_contact.gz'.format(ftype, chrom))
    format_contact(hic, resolution=resolution, chrm=chrom, save_file=True, filename=filename)

def prepare():
    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    chromosomes = ['22']
    resolution = 10000
    for i, chrom in enumerate(chromosomes):
        for j, t in enumerate(['multiple']): # list(replication.keys())
            path = os.path.join('.', t)
            if t == 'multiple':
                name = '4DNFI9PIEPQA.mcool::resolutions/{}'.format(resolution)
            else:
                name = 'hic.cool'
            split_chrom(path, name, t, 'chr'+chrom, resolution)

if __name__ == '__main__':
    prepare()