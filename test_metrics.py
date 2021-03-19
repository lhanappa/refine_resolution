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

raw_list = ['Shen2012-MouseCortex-HindIII-allreps-filtered.10kb.cool', 
        'Selvaraj2013-F123-HindIII-allreps-filtered.10kb.cool',
        'Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
        'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool',
        'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool']

# raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool']

# 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'

methods = ['ours_400', 'low', 'deephic_40', 'hicsr_40']
me_dict = {'deephic_40':'Deephic', 'hicsr_40':'HiCSR', 'ours_400':'EnHiC', 'low':'LR'}
labels = [me_dict[f] for f in methods]

chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']

metrics = ['GenomeDISCO', 'HiC-Spector', 'HiCRep']


data = list()
for cell in raw_list:
    cl = cell.split('-')
    if cl[1] == 'CH12LX' or cl[1] == 'F123' or cl[1] == 'MouseCortex':
        T = 'mm9'
    else:
        T = 'hg19'
    ctype = cl[1]+'_'+cl[2]
    cell_name = '_'.join(cl[0:3]) + '_10kb'
    for chro in chromosomes:
        inpath = os.path.join('.', 'experiment', 'evaluation', cell_name, 
                            'chr{}'.format(chro), 'chromatin_qc', 'scores', 
                            'reproducibility.chr{}.txt'.format(chro))
        if not os.path.exists(inpath):
            continue
        with open(inpath, 'r') as fin:
            for line in fin:
                l = line.split()
                me = l[1]
                if me in methods:
                    # GenomeDISCO, HiC-Spector, HiCRep in order
                    for i, mc in enumerate(metrics):
                        data.append([T, ctype, chro, mc, me, l[i+2]])
        fin.close()

s = pd.DataFrame(data, columns=["type", "cell type", "chromosome", "metric", "method", "value"])

print(s)

output_dir = os.path.join('.', 'experiment', 'evaluation')
for mc in metrics:
    data = s.loc[s['metric']==mc]
    data = data.explode('value')
    data['value'] = data['value'].astype('float')
    
    fig, ax = plt.subplots()
    # ax = sns.catplot(y="cell type", x="value", hue="method", data=data, kind="violin", orient="h", height=12, aspect=.8, width=0.8, scale="width", scale_hue=False)
    g = sns.catplot(ax = ax, y="cell type", x="value", hue="method", hue_order=methods, 
                    data=data, kind="box", orient="h", height=12, aspect=.9)
    # ax.set(xlabel='cell type', ylabel='scores')
    g.set_axis_labels("Score", "Cell type")
    if 'Genome' in mc:
        plt.xlim(-.5, .9)
    else:
        plt.xlim(0.4, 1.0)
    plt.gcf().subplots_adjust(bottom=0.05, top=0.95)
    plt.title('{} scores'.format(mc), size=24)

    # title
    legend_title = 'Method'
    g._legend.set_title(legend_title)
    # replace labels
    for t, l in zip(g._legend.texts, labels): t.set_text(l)

    output = os.path.join(output_dir, 'figure-cell_type')
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'metrics_{}_scores.pdf'.format(mc))
    plt.savefig(output, format='pdf')
