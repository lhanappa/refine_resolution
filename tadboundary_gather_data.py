# gather predict data: predict_[chr*]_10000.npz
# from ./data/[output_ours_2000000_200]/Rao2014_GM12878_10kb/SR to ./experiment/evaluation/
import os
import sys
import shutil
import cooler
import numpy as np
from scipy.sparse import triu 
import pandas as pd

from our_model.utils.operations import remove_zeros, merge_hic, filter_diag_boundary, format_bin, format_contact, sampling_hic
from our_model.utils.operations import scn_normalization, scn_recover
from iced.normalization import ICE_normalization

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def gather(source=None, destination='./experiment/evaluation/', method='output_ours_2000000_200', chromosomes=['19', '20', '21', '22', 'X']):
    if(source is None):
        source = os.path.join('.', 'data', method, 'Rao2014_GM12878_MboI_10kb', 'SR')

    for ch in chromosomes:
        infile = 'predict_chr{}_10000.npz'.format(ch)
        outfile = '{}_predict_chr{}.npz'.format(method, ch)
        inpath = os.path.join(source, infile)
        if os.path.exists(inpath):
            print('copying {} from {} to {}'.format(infile, inpath, os.path.join(destination, 'chr{}'.format(ch), outfile)))
            os.makedirs(os.path.join(
                destination, 'chr{}'.format(ch)), exist_ok=True)
            shutil.copyfile(inpath, os.path.join(
                destination, 'chr{}'.format(ch), outfile))


def gather_high_low_cool(cooler_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', path='./data/raw/', chromosome='22', scale=4, output_path='./experiment/evaluation/'):
    file = os.path.join(path, cooler_file)
    cool_hic = cooler.Cooler(file)
    resolution = cool_hic.binsize
    mat = cool_hic.matrix(balance=True).fetch('chr' + chromosome)
    high_hic, idx = remove_zeros(mat)
    bool_idx = np.array(idx).flatten()
    num_idx = np.array(np.where(idx)).flatten()
    low_hic = sampling_hic(high_hic, scale**2, fix_seed=True)
    print('high hic shape: {}.'.format(high_hic.shape), end=' ')
    print('low hic shape: {}.'.format(low_hic.shape))

    b = {'chrom': ['chr{}'.format(chromosome)]*len(bool_idx), 'start': resolution*np.arange(len(bool_idx)), 'end': resolution*(np.arange(1,(len(bool_idx)+1))), 'weight': 1.0*bool_idx}
    bins = pd.DataFrame(data = b)

    high_hic = triu(high_hic, format='coo')
    low_hic = triu(low_hic, format='coo')

    output_path = os.path.join(output_path, 'chr{}'.format(chromosome))
    os.makedirs(output_path, exist_ok=True)

    outfile = 'high_chr{}.cool'.format(chromosome)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    uri = os.path.join(output_path, outfile)
    p = {'bin1_id': num_idx[high_hic.row], 'bin2_id': num_idx[high_hic.col], 'count': high_hic.data}
    pixels = pd.DataFrame(data = p)
    cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)


    outfile = 'low_chr{}.cool'.format(chromosome)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    uri = os.path.join(output_path, outfile)
    p = {'bin1_id': num_idx[low_hic.row], 'bin2_id': num_idx[low_hic.col], 'count': low_hic.data}
    pixels = pd.DataFrame(data = p)
    cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)



def generate_cool(input_path='./experiment/tad_boundary', chromosomes=['22', '21', '20', '19', 'X'], resolution=10000, genomic_distance=2000000):
    k = np.ceil(genomic_distance/resolution).astype(int)
    for chro in chromosomes:
        path = os.path.join(input_path, 'chr{}'.format(chro))
        hicfile = 'high_chr{}.cool'.format(chro)
        cool_hic = cooler.Cooler(os.path.join(path, hicfile))
        mat = cool_hic.matrix(balance=True).fetch('chr' + chro)
        bins = cool_hic.bins().fetch('chr' + chro)
        num_idx = np.array(np.where(np.array(bins['weight']))).flatten()

        high_mat = mat[num_idx, :]
        high_mat = high_mat[:, num_idx]
        high_mat = filter_diag_boundary(high_mat, diag_k=0, boundary_k=k)

        T = high_mat[600:900, 600:900]
        T = ICE_normalization(high_mat)
        b = {'chrom': ['chr{}'.format(chro)]*T.shape[0], 'start': resolution*np.arange(T.shape[0]), 'end': resolution*np.arange(1, 1+T.shape[0]), 'weight': [1.0]*T.shape[0]}
        bins = pd.DataFrame(data = b)
        coo_mat = triu(T, format='coo')
        # p = {'bin1_id': num_idx[coo_mat.row], 'bin2_id': num_idx[coo_mat.col], 'count': coo_mat.data}
        p = {'bin1_id': coo_mat.row, 'bin2_id': coo_mat.col, 'count': coo_mat.data}
        pixels = pd.DataFrame(data = p)
        uri = os.path.join(path, hicfile)
        cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)

        files = [f for f in os.listdir(path) if '.npz' in f]
        for file in files:
            if 'high' in file or 'low' in file:
                continue
            print(file)
            data = np.load(os.path.join(path, file), allow_pickle=True)
            mat = data['hic']
            namelist = file.split('_')
            if len(namelist) == 3:
                name = namelist[0]
            else:
                model = namelist[1]
                win_len = namelist[3]
                if model == 'hicgan':
                    # true_hic = np.log1p(true_hic)
                    mat = np.expm1(mat)
                elif model == 'deephic':
                    minv = high_mat.min()
                    maxv = high_mat.max()
                    # true_hic = np.divide((true_hic-minv), (maxv-minv), dtype=float,out=np.zeros_like(true_hic), where=(maxv-minv) != 0)
                    mat = mat*(maxv-minv)+minv
                elif model == 'hicsr':
                    log_mat = np.log2(high_mat+1)
                    # ture_hic = 2*(log_mat/np.max(log_mat)) - 1
                    maxv = np.max(log_mat)
                    log_predict_hic = (mat+1)/2*maxv
                    mat = np.expm1(log_predict_hic)
                '''elif model == 'ours':
                    scn, dh = scn_normalization(high_mat, max_iter=3000)
                    mat = scn_recover(mat, dh)'''
                name = '_'.join([model, win_len])
            mat = filter_diag_boundary(mat, diag_k=0, boundary_k=k)
            mat = mat[600:900, 600:900]
            mat = ICE_normalization(mat)
            print('mat shape: {}'.format(mat.shape))
            uri = os.path.join(path, '{}_chr{}.cool'.format(name, chro))
            mat = triu(mat, format='coo')
            # p = {'bin1_id': num_idx[mat.row], 'bin2_id': num_idx[mat.col], 'count': mat.data}
            p = {'bin1_id': mat.row, 'bin2_id': mat.col, 'count': mat.data}
            pixels = pd.DataFrame(data = p)
            cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)
        with open(os.path.join(path, 'track.ini'), 'w' ) as f:
            f.writelines(track)
        f.close()

track = """
[x-axis]
where = top

[x-axis]
fontsize=10

[hic]
file = high_chr22.cool
colormap = Spectral_r
depth = 4000000
min_value = 1
max_value = 800
transform = log1p
file_type = hic_matrix
show_masked_bins = true

[tads]
file = output/high_chr22_domains.bed
file_type = domains
border_color = black
color = none
overlay_previous = share-y
"""


if __name__ == '__main__':
    # methods = ['output_ours_2000000_80', 'output_ours_2000000_200', 'output_ours_2000000_400', 'output_hicsr_2000000_40_28', 'output_deephic_2000000_40_40']
    # methods = ['output_ours_2000000_400', 'output_hicsr_2000000_40_28', 'output_deephic_2000000_40_40']
    methods = ['output_ours_2000000_400']
    
    # cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
    destination_path = os.path.join('.','experiment', 'tad_boundary', cell_type)

    # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    chromosomes = [ '22' ]
    for chro in chromosomes:
        for m in methods:
            source = os.path.join('.', 'data', m, cell_type, 'SR')
            gather(source=source, destination=destination_path, method=m, chromosomes=[chro])
        gather_high_low_cool(cooler_file=cool_file, 
                            path='./data/raw/', 
                            chromosome=chro, 
                            scale=4, 
                            output_path=destination_path)

        generate_cool(input_path=destination_path,
                    chromosomes=[chro],
                    resolution=10000,
                    genomic_distance=2000000)
