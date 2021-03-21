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



    path='./data/raw/'
    for cool_file in raw_list[0:1]:
        # cool_file = 'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool'
        hic = cooler.Cooler(os.path.join(path, cool_file))
        chromosomes = hic.chromnames;
        cnt = 0
        print(hic.info)
        for chro in chromosomes:
            c = hic.matrix(balance=False, sparse=True).fetch(chro)
            cnt += c.sum()
            print(chro, cnt)
        print(cool_file, cnt)

    '''cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_types = [4,8,16,32,48,64]
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
                p = Process(target=plot_demo, args=(source_dir, chro, m, 'x{} downsampling'.format(ct),resolution, start, end, destination_dir))
                queue.append(p)
                p.start()

            for p in queue:
                p.join()'''

