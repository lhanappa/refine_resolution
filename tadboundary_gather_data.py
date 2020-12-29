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

def gather(source=None, destination='./experiment/evaluation/', method='output_ours_2000000_200', chromosomes=['19', '20', '21', '22', 'X']):
    if(source is None):
        source = os.path.join('.', 'data', method, 'Rao2014_GM12878_MboI_10kb', 'SR')

    for ch in chromosomes:
        infile = 'predict_chr{}_10000.npz'.format(ch)
        outfile = '{}_predict_chr{}_10000.npz'.format(method, ch)
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
    # resolution = cool_hic.binsize
    mat = cool_hic.matrix(balance=True).fetch('chr' + chromosome)
    high_hic, idx = remove_zeros(mat)
    low_hic = sampling_hic(high_hic, scale**2, fix_seed=True)


    b = {'chrom': ['chr{}'.format(chromosome)]*mat.shape[0], 'start': resolution*idx, 'end': resolution*(idx+1), 'weight': [1.0]*mat.shape[0]}
    bins = pd.DataFrame(data = b)
    print(bins)

    high_hic = triu(high_hic, format='coo')
    low_hic = triu(low_hic, format='coo')

    output_path = os.path.join(output_path, 'chr{}'.format(chromosome))
    os.makedirs(output_path, exist_ok=True)

    outfile = 'high_chr{}_10000.cool'.format(chromosome)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    uri = os.path.join(output_path, outfile)
    p = {'bin1_id': high_hic.row, 'bin2_id': high_hic.col, 'count': high_hic.data}
    pixels = pd.DataFrame(data = p)
    cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)


    outfile = 'low_chr{}_{}0000.cool'.format(chromosome, scale)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    uri = os.path.join(output_path, outfile)
    p = {'bin1_id': low_hic.row, 'bin2_id': low_hic.col, 'count': low_hic.data}
    pixels = pd.DataFrame(data = p)
    cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)



def generate_cool(input_path='./experiment/tad_boundary', chromosomes=['22', '21', '20', '19', 'X'], resolution=10000, genomic_distance=2000000):
    k = np.ceil(genomic_distance/resolution).astype(int)
    for chro in chromosomes:
        path = os.path.join(input_path, 'chr{}'.format(chro))
        hicfile = 'high_chr{}_10000.cool'.format(chro)
        cool_hic = cooler.Cooler(os.path.join(path, hicfile))
        mat = cool_hic.matrix(balance=True).fetch('chr' + chro)
        mat = filter_diag_boundary(mat, diag_k=2, boundary_k=k)

        bins = cool_hic.bins().fetch('chr' + chro)
        high_mat = mat

        files = [f for f in os.listdir(path) if '.npz' in f]
        for file in files:
            if 'high' in file:
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
            mat = filter_diag_boundary(mat, diag_k=2, boundary_k=k)

            print('mat shape: {}'.format(mat.shape))
            uri = os.path.join(path, '{}.cool'.format(name))
            mat = triu(mat, format='coo')
            p = {'bin1_id': mat.row, 'bin2_id': mat.col, 'count': mat.data}
            pixels = pd.DataFrame(data = p)
            cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)




if __name__ == '__main__':
    # methods = ['output_ours_2000000_80', 'output_ours_2000000_200', 'output_ours_2000000_400', 'output_hicsr_2000000_40_28', 'output_deephic_2000000_40_40']
    # methods = ['output_ours_2000000_400', 'output_hicsr_2000000_40_28', 'output_deephic_2000000_40_40']
    mathods = ['output_ours_2000000_400']
    
    # cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
    destination_path = os.path.join('.','experiment', 'tad_boundary', cell_type)

    for m in methods:
        source = os.path.join('.', 'data', m, cell_type, 'SR')
        gather(source=source, destination=destination_path, method=m)

    # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    chromosomes = ['22']
    for chro in chromosomes:
        gather_high_low_cool(cooler_file=cool_file, 
                            path='./data/raw/', 
                            chromosome=chro, 
                            scale=4, 
                            output_path=destination_path)

        generate_cool(input_path=destination_path,
                    chromosomes=chro,
                    resolution=10000,
                    genomic_distance=2000000)
