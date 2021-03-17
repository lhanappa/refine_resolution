import os, sys, shutil, gzip
import numpy as np
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

'''raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
        'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool']'''

raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool']
depth_ratio = [4,8,16,32,48,64]
# 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'

methods = ['ours_400']
me_dict = {'deephic_40':'Deephic', 'hicsr_40':'HiCSR', 'ours_400':'EnHiC', 'low':'LR'}
labels = [me_dict[f] for f in methods]

chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']

metrics = ['GenomeDISCO', 'HiC-Spector', 'HiCRep']


data = list()
hic_ = raw_list[0]
cl = hic_.split('-')
ctype = cl[1]+'_'+cl[2]
for dr in depth_ratio:
    ratio_name = '_'.join(cl[0:3]) +'-' + str(dr) + '_10kb'
    for chro in chromosomes:
        inpath = os.path.join('.', 'experiment', 'seq_depth_ratio', ratio_name, 
                            'chr{}'.format(chro), 'chromatin_qc', 'scores', 
                            'reproducibility.chr{}.txt'.format(chro))
        print(inpath)
        if not os.path.exists(inpath):
            continue
        with open(inpath, 'r') as fin:
            for line in fin:
                print(line)
                l = line.split()
                me = l[1]
                if me in methods:
                    # GenomeDISCO, HiC-Spector, HiCRep in order
                    for i, mc in enumerate(metrics):
                        data.append([dr, chro, mc, l[i+2]])
        fin.close()

s = pd.DataFrame(data, columns=["ratio", "chromosome", "metric", "value"])

print(s)

output_dir = os.path.join('.', 'experiment', 'seq_depth_ratio')
for mc in metrics:
    data = s.loc[s['metric']==mc]
    data = data.explode('value')
    data['value'] = data['value'].astype('float')
    
    fig, ax = plt.subplots()
    # ax = sns.catplot(y="cell type", x="value", hue="method", data=data, kind="violin", orient="h", height=12, aspect=.8, width=0.8, scale="width", scale_hue=False)
    # g = sns.catplot(ax = ax, y="cell type", x="value", hue="method", hue_order=methods, data=data, kind="box", orient="h", height=12, aspect=.9)
    g = sns.lineplot(ax=ax, data=data, x="chromosome", y="value", hue="ratio")

    # ax.set(xlabel='cell type', ylabel='scores')
    '''g.set_axis_labels("Score", "Cell type")
    if 'Genome' in mc:
        plt.xlim(-.5, .9)
    else:
        plt.xlim(0.4, 1.0)'''
    plt.gcf().subplots_adjust(bottom=0.05, top=0.95)
    plt.title('{} scores'.format(mc), size=24)

    # title
    '''legend_title = 'Method'
    g._legend.set_title(legend_title)
    # replace labels
    for t, l in zip(g._legend.texts, labels): t.set_text(l)'''

    output = os.path.join(output_dir, 'figure-seq_depth')
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'metrics_{}_scores.pdf'.format(mc))
    plt.savefig(output, format='pdf')
