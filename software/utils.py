import os
import numpy as np
import cooler


def redircwd_back_projroot(project_name='refine_resolution'):
    root = os.getcwd().split('/')
    for i, f in enumerate(root):
        if f == project_name:
            root = root[:i+1]
            break
    root = '/'.join(root)
    os.chdir(root)
    print('current working directory: ', os.getcwd())
    return root


def remove_zeros(matrix):
    idxy = ~np.all(np.isnan(matrix), axis=0)
    M = matrix[idxy, :]
    M = M[:, idxy]
    M = np.asarray(M)
    idxy = np.asarray(idxy)
    return M, idxy


def cool_to_raw(raw_path, raw_hic):
    file = os.path.join(raw_path, raw_hic)
    print('raw hic data: ', file)
    cool_hic = cooler.Cooler(file)
    resolution = cool_hic.binsize
    return cool_hic, resolution


def configure(raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
              genomic_distance=2000000,
              lr_size=40,
              hr_size=40,
              downsample_factor=16,
              methods_name='hicsr',
              ):

    root_dir = redircwd_back_projroot(project_name='refine_resolution')
    experiment_name = '_'.join(
        [methods_name, str(genomic_distance), str(lr_size), str(hr_size)])

    data_cat = raw_hic.split(
        '-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('.')[1]
    input_path = os.path.join(
        root_dir, 'data', 'input_'+experiment_name, data_cat)+'/'
    if methods_name == 'hicsr':
        script_work_dir = os.path.join(root_dir, 'software', 'HiCSR')
    elif methods_name == 'deephic':
        script_work_dir = os.path.join(root_dir, 'software', 'DeepHiC')

    return raw_hic, genomic_distance, lr_size, hr_size, downsample_factor, \
        root_dir, experiment_name, input_path, script_work_dir


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
