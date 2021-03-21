import os, sys, shutil, gzip
import numpy as np
from scipy.sparse import coo_matrix, triu
from scipy.spatial import distance
from scipy import stats
import subprocess
from multiprocessing import Process
import pandas as pd

import cooler
from iced import filter
from iced import normalization

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning, DeprecationWarning, RuntimeWarning))
# using fithic to find significant interactions by CLI

from our_model.utils.operations import remove_zeros, sampling_hic
from our_model.utils.operations import scn_normalization, scn_recover
from iced.normalization import ICE_normalization

def filter_diag_boundary(hic, diag_k=1, boundary_k=None):
    if boundary_k is None:
        boundary_k = hic.shape[0]-1
    boundary_k = min(hic.shape[0]-1, boundary_k)
    filter_m = np.tri(N=hic.shape[0], k=boundary_k)
    filter_m = np.triu(filter_m, k=diag_k)
    filter_m = filter_m + np.transpose(filter_m)
    return np.multiply(hic, filter_m)

def plot_demo(source_dir, chromosome, model_name, resolution, start, end, destination_dir):
    cool_file = os.path.join(source_dir, 'sample_{}_chr{}.cool'.format(model_name, chromosome))
    hic = cooler.Cooler(cool_file)
    start = max(0, int(start))
    if end > hic.chromsizes['chr{}'.format(chromosome)]:
        length = end - start
        end = hic.chromsizes['chr{}'.format(chromosome)]
        start = end - length

    region = ('chr{}'.format(chromosome), start, end)
    hic_mat = hic.matrix(balance=True).fetch(region)
    hic_mat = normalization.ICE_normalization(hic_mat)
    # hic_bins = hic.bins().fetch(region)
    # weight = hic_bins['weight']
    # filter_idx = np.array(np.where(weight==1)).flatten()


    fig, ax0 = plt.subplots()
    cmap = plt.get_cmap('RdBu_r')
    hic_mat = filter_diag_boundary(hic_mat, diag_k=1, boundary_k=200)
    Z = np.log1p(hic_mat)

    bounds = np.append(np.arange(0,7,0.06), np.arange(7,12,0.3))
    norm = matplotlib.colors.BoundaryNorm(boundaries=bounds, ncolors=256)
    im = ax0.imshow(Z, cmap=cmap, norm=norm) #, vmin=0, vmax=8
    fig.colorbar(im, ax=ax0, ticks=np.arange(0,8))

    legend = {'ours': 'EnHiC', 'deephic': 'Deephic', 'hicsr':'HiCSR', 'low':'LR', 'high':'HR'}
    name = model_name.split('_')[0]
    ax0.set_title('{} log1p Scale'.format(legend[name]))
    ax0.set_xlim(-1, hic_mat.shape[0])
    ax0.set_ylim(-1, hic_mat.shape[1])
    fig.tight_layout()
    output = destination_dir
    os.makedirs(output, exist_ok=True)
    plt.savefig(os.path.join(output, 'demo_{}.pdf'.format(legend[name])), format='pdf')
    plt.savefig(os.path.join(output, 'demo_{}.jpg'.format(legend[name])), format='jpg')

def generate_cool(input_path='./experiment/significant_interactions', chromosomes=['22', '21', '20', '19', 'X'], resolution=10000, genomic_distance=2000000):
    k = np.ceil(genomic_distance/resolution).astype(int)
    for chro in chromosomes:
        path = os.path.join(input_path, 'chr{}'.format(chro))
        hicfile = 'sample_high_chr{}.cool'.format(chro)
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
            uri = os.path.join(path, 'sample_{}_chr{}.cool'.format(name, chro))
            mat = triu(mat, format='coo')
            # p = {'bin1_id': mat.row, 'bin2_id': mat.col, 'count': mat.data}
            p = {'bin1_id': num_idx[mat.row], 'bin2_id': num_idx[mat.col], 'count': mat.data}
            pixels = pd.DataFrame(data = p)
            cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)

def gather_high_low_cool(cooler_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', path='./data/raw/', chromosome='22', scale=16, output_path='./experiment/significant_interactions/'):
    file = os.path.join(path, cooler_file)
    cool_hic = cooler.Cooler(file)
    resolution = cool_hic.binsize
    mat = cool_hic.matrix(balance=True).fetch('chr' + chromosome)
    high_hic, idx = remove_zeros(mat)
    bool_idx = np.array(idx).flatten()
    num_idx = np.array(np.where(idx)).flatten()
    low_hic = sampling_hic(high_hic, scale, fix_seed=True)
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

    outfile = 'sample_high_chr{}.cool'.format(chromosome)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    uri = os.path.join(output_path, outfile)
    p = {'bin1_id': num_idx[high_hic.row], 'bin2_id': num_idx[high_hic.col], 'count': high_hic.data}
    pixels = pd.DataFrame(data = p)
    cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)

    outfile = 'sample_low_chr{}.cool'.format(chromosome)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    uri = os.path.join(output_path, outfile)
    p = {'bin1_id': num_idx[low_hic.row], 'bin2_id': num_idx[low_hic.col], 'count': low_hic.data}
    pixels = pd.DataFrame(data = p)
    cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)

if __name__ == '__main__':
    raw_list = ['Rao2014-GM12878-MboI-allreps-filtered.10kb.cool',
            'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
            'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
            'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
            'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
            'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
            'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
            'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool']


    '''
    # 'Shen2012-MouseCortex-HindIII-allreps-filtered.10kb.cool', 
    # 'Selvaraj2013-F123-HindIII-allreps-filtered.10kb.cool',
    # 'Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
    raw_list = [
            'Selvaraj2013-F123-HindIII-allreps-filtered.10kb.cool',
            'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
            'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
            'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool']'''

    methods = ['ours_400', 'low', 'deephic_40', 'hicsr_40']
    me_dict = {'deephic_40':'Deephic', 'hicsr_40':'HiCSR', 'ours_400':'EnHiC', 'low':'LR'}
    labels = [me_dict[f] for f in methods]

    chromosomes = ['19']
    [start, end] = [1400*10000, 1600*10000]

    genome_dis = int(100)
    window_len = int(200)

    '''for cool_file in raw_list:
        # cool_file = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
        cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
        hic_info = cooler.Cooler(os.path.join('.', 'data', 'raw', cool_file))
        resolution = int(hic_info.binsize) # 10000, 10kb

        # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
        # chromosomes = [str(sys.argv[1])] # ['17', '18', '19', '20', '21', '22', 'X']# [str(sys.argv[1])]
        # [start, end] = [int(sys.argv[2]), int(sys.argv[3])]

        destination_path = os.path.join('.','experiment', 'evaluation', cell_type)
        for chro in chromosomes:
            gather_high_low_cool(cooler_file=cool_file, 
                                path='./data/raw/', 
                                chromosome=chro, 
                                scale=16, 
                                output_path=destination_path)

            generate_cool(input_path=destination_path,
                        chromosomes=[chro],
                        resolution=10000,
                        genomic_distance=2000000)

            path = os.path.join('.', 'experiment', 'evaluation', cell_type, 'chr{}'.format(chro))
            files = [f for f in os.listdir(path) if '.cool' in f]
            queue = []
            print(start, end)
            source_dir = path
            for file in files:
                m = file.split('_')[1:-1]
                m = '_'.join(m)

                # plot_significant_interactions(source_dir, chro, m, resolution, low_dis=low, up_dis=up, start=start, end=end)
                destination_dir = os.path.join('.', 'experiment', 'evaluation', 'figure_sample', '{}_{}'.format(start, end), cell_type)
                p = Process(target=plot_demo, args=(source_dir, chro, m, resolution, start, end, destination_dir))
                queue.append(p)
                p.start()

            for p in queue:
                p.join()'''

    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_types = [4,8,16,32,48]
    for ct in cell_types:
        cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] +'-' + str(ct) + '_' + cool_file.split('.')[1]
        hic_info = cooler.Cooler(os.path.join('.', 'data', 'raw', cool_file))
        resolution = int(hic_info.binsize) # 10000, 10kb

        # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
        # chromosomes = [str(sys.argv[1])] # ['17', '18', '19', '20', '21', '22', 'X']# [str(sys.argv[1])]
        # [start, end] = [int(sys.argv[2]), int(sys.argv[3])]

        destination_path = os.path.join('.','experiment', 'seq_depth_ratio', cell_type)
        for chro in chromosomes:
            gather_high_low_cool(cooler_file=cool_file, 
                                path='./data/raw/', 
                                chromosome=chro, 
                                scale=ct, 
                                output_path=destination_path)

            generate_cool(input_path=destination_path,
                        chromosomes=[chro],
                        resolution=10000,
                        genomic_distance=2000000)

            path = os.path.join('.', 'experiment', 'seq_depth_ratio', cell_type, 'chr{}'.format(chro))
            files = [f for f in os.listdir(path) if '.cool' in f]
            queue = []
            print(start, end)
            source_dir = path
            for file in files:
                m = file.split('_')[1:-1]
                m = '_'.join(m)

                # plot_significant_interactions(source_dir, chro, m, resolution, low_dis=low, up_dis=up, start=start, end=end)
                destination_dir = os.path.join('.', 'experiment', 'seq_depth_ratio', 'figure_sample', '{}_{}'.format(start, end), cell_type)
                p = Process(target=plot_demo, args=(source_dir, chro, m, resolution, start, end, destination_dir))
                queue.append(p)
                p.start()

            for p in queue:
                p.join()

