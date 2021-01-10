# gather predict data: predict_[chr*]_10000.npz
# from ./data/[output_ours_2000000_200]/Rao2014_GM12878_10kb/SR to ./experiment/evaluation/
import os
import sys
import shutil
import cooler
import numpy as np
from scipy.sparse import triu, coo_matrix

import pandas as pd

from our_model.utils.operations import remove_zeros, merge_hic, filter_diag_boundary, format_bin, format_contact, sampling_hic
from our_model.utils.operations import scn_normalization, scn_recover
from iced.normalization import ICE_normalization

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def gather(source=None, destination='./experiment/significant_interactions/', method='output_ours_2000000_200', chromosomes=['19', '20', '21', '22', 'X']):
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
        else:
            print('not exist {} {}'.format(infile, inpath))


def gather_high_low_cool(cooler_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', path='./data/raw/', chromosome='22', scale=4, output_path='./experiment/significant_interactions/'):
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

    # high_hic = ICE_normalization(high_hic)
    # low_hic = ICE_normalization(low_hic)

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



def generate_cool(input_path='./experiment/significant_interactions', chromosomes=['22', '21', '20', '19', 'X'], resolution=10000, genomic_distance=2000000):
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
                    mat = (mat+np.transpose(mat))/2
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
            # mat = ICE_normalization(mat)
            print('{} matrix shape: {}'.format(name, mat.shape))
            uri = os.path.join(path, '{}_chr{}.cool'.format(name, chro))
            mat = triu(mat, format='coo')
            # p = {'bin1_id': mat.row, 'bin2_id': mat.col, 'count': mat.data}
            p = {'bin1_id': num_idx[mat.row], 'bin2_id': num_idx[mat.col], 'count': mat.data}
            pixels = pd.DataFrame(data = p)
            cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)


def generate_fragments(chromosome, matrix, bins, output):
    # bins pandas dataframe, 
    chro_name = str(chromosome[3:])
    hit_count = (matrix.sum(axis=0)).flatten()
    mid_points = int( (bins[:, 1] + bins[:, 2])/2 )
    with open(os.path.join(output+'_fragments.txt')) as f:
        for i, mp in enumerate(mid_points):
            line = '{}\t0\t{}\t{}\t0\n'.format(chro_name, mp, hit_count[i])
            f.write(line)
    f.close()


def generate_interactions(chromosome, matrix, bins, output):
    chro_name = str(chromsome[3:])
    mid_points = int( (bins[:, 1] + bins[:, 2])/2 )
    mat = int(matrix)
    coo_data = coo_matrix(mat)
    idx1 = mid_points[coo_data.row]
    idx2 = mid_points[coo_data.col]
    data = coo_data.data
    with open(os.path.join(output+'_interactions.txt')) as f:
        for i, mp in enumerate(mid_points):
            line = '{}\t{}\t{}\t{}\t{}\n'.format(chro_name, idx1, chro_name, idx2, data)
            f.write(line)
    f.close()

def geneate_biases_ICE():
    pass

def generate_fithic_files(cool_file, chromosome, start, end, output):
    hic = cooler.Cooler(cool_file)
    region = ('chr{}'.format(chromosome), start, end)
    hic_mat = hic.matrix(balance=True).fetch(region)
    hic_bins = hic.bins().fetch(region)

    hic_bins = hic_bins.to_numpy()
    generate_fragments(chromosome, hic_mat, hic_bins, output)
    generate_interactions(chromosome, hic_mat, bins, output)




if __name__ == '__main__':
    # methods = ['output_ours_2000000_80', 'output_ours_2000000_200', 'output_ours_2000000_400', 'output_hicsr_2000000_40_28', 'output_deephic_2000000_40_40']
    methods = ['output_ours_2000000_400', 'output_hicsr_2000000_40_28', 'output_deephic_2000000_40_40']
    # methods = [str(sys.argv[1])]

    # cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
    destination_path = os.path.join('.','experiment', 'significant_interactions', cell_type)

    # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    # chromosomes = [ '22' ]
    chromosomes = [str(sys.argv[1])]
    [start, end] = [45000000, 45100000]
    for chro in chromosomes:
        # for m in methods:
            # source = os.path.join('.', 'data', m, cell_type, 'SR')
            # gather(source=source, destination=destination_path, method=m, chromosomes=[chro])

        gather_high_low_cool(cooler_file=cool_file, 
                            path='./data/raw/', 
                            chromosome=chro, 
                            scale=4, 
                            output_path=destination_path)

        generate_cool(input_path=destination_path,
                    chromosomes=[chro],
                    resolution=10000,
                    genomic_distance=2000000)

        path = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro))
        files = [f for f in os.listdir(path) if '.cool' in f]
        for file in files:
            m = file.split('.')[0]
            source = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro), file)
            dest =  os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro), 'output')
            os.makedirs(dest, exist_ok=True)
            dest = os.path.join(dest, m)
            generate_fithic_files(source, chro, start, end, output=dest)
