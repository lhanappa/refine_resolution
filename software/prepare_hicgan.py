import os
import sys
import cooler
import numpy as np
import multiprocessing
from .utils import redircwd_back_projroot, cool_to_raw
from .utils import remove_zeros, sampling_hic


def save_to_compressed(hic, idx, output_path, output_name):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    output = os.path.join(output_path, output_name)
    hic = np.asarray(hic)
    print(output)
    np.savez_compressed(output, hic=hic, compact=idx)

# deviding method


def divide(mat, chr_num, chunk_size=40, stride=28, bound=201, padding=True, verbose=False):
    chr_str = str(chr_num)
    result = []
    index = []
    size = mat.shape[0]
    if (stride < chunk_size and padding):
        pad_len = (chunk_size - stride) // 2
        mat = np.pad(mat, ((pad_len, pad_len), (pad_len, pad_len)), 'constant')
    # mat's shape changed, update!
    height, width = mat.shape
    assert height == width, 'Now, we just assumed matrix is squared!'
    for i in range(0, height, stride):
        for j in range(0, width, stride):
            if abs(i-j) <= bound and i+chunk_size < height and j+chunk_size < width:
                subImage = mat[i:i+chunk_size, j:j+chunk_size]
                result.append([subImage])
                index.append((chr_num, size, i, j))
    result = np.array(result)
    if verbose:
        print(f'[Chr{chr_str}] Deviding HiC matrix ({size}x{size}) into {len(result)} samples with chunk={chunk_size}, stride={stride}, bound={bound}')
    index = np.array(index)
    return result, index


def hicgan_divider(n, high_file, down_file, scale=1, chunk=40, stride=40, bound=201):
    hic_data = np.load(high_file)
    down_data = np.load(down_file)
    full_size = hic_data['hic'].shape[0]
    # Compacting
    #hic = compactM(hic_data['hic'], compact_idx)
    #down_hic = compactM(down_data['hic'], compact_idx)
    hic = hic_data['hic']
    down_hic = down_data['hic']

    # Deviding and Pooling (pooling is not performed actually)
    div_dhic, div_inds = divide(down_hic, n, chunk, stride, bound)
    #div_dhic = pooling(div_dhic, scale, pool_type=pool_type, verbose=False).numpy()
    div_hhic, _ = divide(hic, n, chunk, stride, bound, verbose=True)
    return n, div_dhic, div_hhic, div_inds, full_size


def run(raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        chromosome_list=['22'],
        genomic_distance=2000000,
        lr_size=40,
        hr_size=40,
        downsample_factor=16):
    methods_name = 'hicgan'
    root_dir = redircwd_back_projroot(project_name='refine_resolution')
    experiment_name = '_'.join(
        [methods_name, str(genomic_distance), str(lr_size), str(hr_size)])
    data_cat = raw_hic.split(
        '-')[0] + '_' + raw_hic.split('-')[1] + '_' + raw_hic.split('.')[1]
    input_path = os.path.join(
        root_dir, 'data', 'input_'+experiment_name, data_cat)
    # preprocessing_output_path = os.path.join( root_dir, 'data', 'input_' + experiment_name, data_cat)

    cell_type = raw_hic.split('-')[1]

    [hic_m, hi_res] = cool_to_raw(raw_path=os.path.join(
        root_dir, 'data', 'raw'), raw_hic=raw_hic)

    low_res = int(np.sqrt(downsample_factor)*hi_res)

    for chro in chromosome_list:
        # chrN_10kb.npz
        name_hr = f'chr{chro}_{hi_res}.npz'
        chromosome = 'chr' + chro
        mat_hr = hic_m.matrix(balance=True).fetch(chromosome)
        [mat_hr, idx] = remove_zeros(mat_hr)

        # chrN_40kb.npz
        name_lr = f'chr{chro}_{low_res}.npz'
        mat_lr = sampling_hic(mat_hr, downsample_factor, fix_seed=True)

        # normalization log1p
        mat_hr = np.log1p(mat_hr)
        mat_lr = np.log1p(mat_lr)
        save_to_compressed(mat_hr, idx, output_path=os.path.join(
            input_path, 'hr'), output_name=name_hr)
        save_to_compressed(mat_lr, idx, output_path=os.path.join(
            input_path, 'lr'), output_name=name_lr)


if __name__ == '__main__':
    run()
