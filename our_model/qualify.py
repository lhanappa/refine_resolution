import time
import datetime
import cooler
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

from iced import normalization
from utils import operations, quality_hic
# 'Dixon2012-H1hESC-HindIII-allreps-filtered.10kb.cool'


def configure_our_model(
        path='./data',
        raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        sr_path = 'output',
        chromosome='22',
        genomic_distance=2000000,
        resolution=10000):

    sr_file = raw_file.split('-')[0] + '_' + raw_file.split('.')[1]
    directory_sr = os.path.join(path, sr_path, sr_file, 'SR', 'chr'+chromosome)

    input_path = directory_sr
    input_file = sr_file+'_chr'+chromosome+'.npz'
    if not os.path.exists(os.path.join(input_path, input_file)):
        print('not input file')

    print('input path: ', input_path, input_file)
    with np.load(os.path.join(input_path, input_file), allow_pickle=True) as data:
        predict_hic = data['predict_hic']
        true_hic = data['true_hic']
        idx_1d_2d = data['index_1D_2D'][()]  # get dict()
        idx_2d_1d = data['index_2D_1D'][()]  # get dict()
        start = data['start_id']
        end = data['end_id']

    print(end+1)
    print(idx_1d_2d)

    if genomic_distance is None:
        max_boundary = None
    else:
        max_boundary = np.ceil(genomic_distance/(resolution))

    predict_hic_hr_merge = operations.merge_hic(
        predict_hic, index_1D_2D=idx_1d_2d, max_distance=max_boundary)
    print('shape of merge predict hic hr', predict_hic_hr_merge.shape)

    true_hic_hr_merge = operations.merge_hic(
        true_hic, index_1D_2D=idx_1d_2d, max_distance=max_boundary)
    print('shape of merge predict hic hr', predict_hic_hr_merge.shape)

    k = np.ceil(genomic_distance/resolution).astype(int)
    true_hic_hr_merge = operations.filter_diag_boundary(
        true_hic_hr_merge, diag_k=2, boundary_k=k)
    predict_hic_hr_merge = operations.filter_diag_boundary(
        predict_hic_hr_merge, diag_k=2, boundary_k=k)

    print('sum true:', np.sum(np.abs(true_hic_hr_merge)))
    print('sum predict:', np.sum(np.abs(predict_hic_hr_merge)))
    diff = np.abs(true_hic_hr_merge-predict_hic_hr_merge)
    print('sum diff: {:.5}'.format(np.sum(diff**2)))

    '''fig, axs = plt.subplots(1, 2, figsize=(8, 15))
    ax = axs[0].imshow(np.log1p(1000*predict_hic_hr_merge), cmap='RdBu_r')
    axs[0].set_title('predict')
    ax = axs[1].imshow(np.log1p(1000*true_hic_hr_merge), cmap='RdBu_r')
    axs[1].set_title('true')
    plt.tight_layout()
    plt.show()'''

    operations.format_bin(true_hic_hr_merge, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'.bed.gz')
    operations.format_contact(true_hic_hr_merge, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'_contact_true.gz')
    operations.format_contact(predict_hic_hr_merge, coordinate=(
        0, 1), resolution=10000, chrm=chromosome, save_file=True, filename=input_path+'/'+ sr_file+'_contact_predict.gz')

    return input_path, sr_file


def score_hicrep(file1,
                 file2,
                 bedfile,
                 output_path,
                 script='./utils/hicrep_wrapper.R',
                 maxdist=int(2000000),
                 resolution=int(10000),
                 h=int(20),
                 m1name='m1',
                 m2name='m2'):
    quality_hic.run_hicrep(script=script, f1=file1, f2=file2,
                           bedfile=bedfile, output_path=output_path, maxdist=maxdist,
                           resolution=resolution,
                           h=h,
                           m1name=m1name,
                           m2name=m2name)


if __name__ == '__main__':
    root = operations.redircwd_back_projroot(project_name='refine_resolution')
    [input_path, hicfile] = configure_our_model(path = os.path.join(root, 'data'))
    file1 = '"' + input_path+'/' + hicfile + '_contact_true.gz' + '"'
    file2 = '"' +input_path+'/' + hicfile + '_contact_predict.gz'+ '"'
    output = '"' +input_path+'/'+ hicfile + '_scores.txt'+ '"'
    bedfile = '"' +input_path+'/'+ hicfile+ '.bed.gz' + '"'
    script = './our_model/utils/hicrep_wrapper.R'
    score_hicrep(script=script, file1=file1, file2=file2,
                 bedfile=bedfile, output_path=output)
