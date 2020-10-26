# gather predict data: predict_[chr*]_10000.npz
# from ./data/[output_ours_2000000_200]/Rao2014_GM12878_10kb/SR to ./experiment/evaluation/
import os, sys
import shutil
import cooler

from our_model.utils.operations import remove_zeros, merge_hic, filter_diag_boundary, format_bin, format_contact, sampling_hic

def gather(source=None, destination='./experiment/evaluation/', method='ours', chromosomes=['19', '20','21', '22', 'X']):
    if(source is None):
        source = os.path.join('data', 'output_{}_2000000_200', 'Rao2014_GM12878_10kb','SR'.format(method))
    os.makedirs(destination, exist_ok=True)

    for ch in chromosomes:
        infile = 'predict_chr{}_10000.npz'.format(ch)
        outfile = '{}_predict_chr{}_10000.npz'.format(method, ch)
        inpath = os.path.join(source, infile)
        if os.path.exists(inpath):
            shutil.copyfile(inpath, os.path.join(destination, outfile))


def gather_high_low_mat(cooler_file, path, chromosome, scale=4, output_path = './experiment/evaluation/'):
    file = os.path.join(path, cooler_file)
    cool_hic = cooler.Cooler(file)
    # resolution = cool_hic.binsize
    mat = cool_hic.matrix(balance=True).fetch('chr' + chromosome)
    high_hic, idx = remove_zeros(mat)
    low_hic = sampling_hic(high_hic, scale**2, fix_seed=True)

    outfile = 'high_chr{}_10000.npz'.format(chromosome)
    np.savez_compressed(os.path.join(output_path, outfile), hic=high_hic, compact=idx)
    outfile = 'low_chr{}_{}0000.npz'.format(chromosome, scale)
    np.savez_compressed(os.path.join(output_path, outfile), hic=low_hic, compact=idx)
