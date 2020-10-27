# gather predict data: predict_[chr*]_10000.npz
# from ./data/[output_ours_2000000_200]/Rao2014_GM12878_10kb/SR to ./experiment/evaluation/
import os
import sys
import shutil
import cooler
import numpy as np

from our_model.utils.operations import remove_zeros, merge_hic, filter_diag_boundary, format_bin, format_contact, sampling_hic


def gather(source=None, destination='./experiment/evaluation/', method='output_ours_2000000_200', chromosomes=['19', '20', '21', '22', 'X']):
    if(source is None):
        source = os.path.join('.', 'data', method,
                              'Rao2014_GM12878_10kb', 'SR')

    for ch in chromosomes:
        infile = 'predict_chr{}_10000.npz'.format(ch)
        outfile = '{}_predict_chr{}_10000.npz'.format(method, ch)
        inpath = os.path.join(source, infile)
        if os.path.exists(inpath):
            print('copying {} from {} to {}'.format(infile, inpath, os.path.join(destination, 'chr{}'.format(ch), outfile)))
            os.makedirs(os.path.join( destination, 'chr{}'.format(ch)), exist_ok=True)
            shutil.copyfile(inpath, os.path.join( destination, 'chr{}'.format(ch), outfile))


def gather_high_low_mat(cooler_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', path='./data/raw/', chromosome='22', scale=4, output_path='./experiment/evaluation/'):
    file = os.path.join(path, cooler_file)
    cool_hic = cooler.Cooler(file)
    # resolution = cool_hic.binsize
    mat = cool_hic.matrix(balance=True).fetch('chr' + chromosome)
    high_hic, idx = remove_zeros(mat)
    low_hic = sampling_hic(high_hic, scale**2, fix_seed=True)

    output_path = os.path.join(output_path, 'chr{}'.format(chromosome))
    os.makedirs(output_path, exist_ok=True)

    outfile = 'high_chr{}_10000.npz'.format(chromosome)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    np.savez_compressed(os.path.join(output_path, outfile),
                        hic=high_hic, compact=idx)
    outfile = 'low_chr{}_{}0000.npz'.format(chromosome, scale)
    print('saving file {}'.format(os.path.join(output_path, outfile)))
    np.savez_compressed(os.path.join(output_path, outfile),
                        hic=low_hic, compact=idx)


def generate_bin(mat, chromosome, output_path, filename='bins', resolution=10000):
    format_bin(mat,
               coordinate=(0, 1),
               resolution=resolution,
               chrm=chromosome,
               save_file=True,
               filename=os.path.join(output_path, '{}_chr{}.bed.gz'.format(filename, chromosome))
               )


def generate_coo(mat, chromosome, output_path, filename, genomic_distance, resolution=10000):
    k = np.ceil(genomic_distance/resolution).astype(int)
    filt_mat = filter_diag_boundary(mat, diag_k=2, boundary_k=k)
    format_contact(filt_mat, 
                    coordinate=(0, 1), 
                    resolution=resolution, 
                    chrm=chromosome, 
                    save_file=True, 
                    filename=os.path.join(output_path, '{}_chr{}_contact.gz'.format(filename, chromosome))
               )

def generate_prefile(input_path='./experiment/evaluation', chromosomes = ['22','21','20','19','X'], genomic_distance=2000000):
    for chro in chromosomes:
        path = os.path.join(input_path, 'chr{}'.format(chro))
        files = [f for  f in os.listdir(path) if '.npz' in f]
        for file in files:
            print(file)
            data = np.load(os.path.join(path, file), allow_pickle=True)
            mat = data['hic']
            namelist = file.split('_')
            if len(namelist) == 3:
                name = namelist[0]
            else:
                name = namelist[1]
            generate_coo(mat, chromosome=chro, output_path=path, filename=name, genomic_distance=genomic_distance)
            if 'high' in file:
                generate_bin(mat, chromosome=chro, output_path=path)
