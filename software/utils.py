import os
import numpy as np


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
