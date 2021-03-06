import os, sys, shutil, gzip
import numpy as np
import pandas as pd

import cooler
from iced import filter
from iced import normalization

from matplotlib import pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

import cooler

import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning, DeprecationWarning, RuntimeWarning))
# using fithic to find significant interactions by CLI
def ttest_greater(a, b):
    #For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    N = len(a)
    var_a = np.array(a).var(ddof=1)
    var_b = np.array(b).var(ddof=1)
    # std deviation
    s = np.sqrt((var_a + var_b)/2)
    ## Calculate the t-statistics
    t = (a.mean() - b.mean())/(s*np.sqrt(2/N))
    ## Compare with the critical t-value
    # Degrees of freedom
    df = 2*N - 2
    # p-value after comparison with the t 
    p = 1 - stats.t.cdf(t,df=df)
    print("t = " + str(t))
    print("p = " + str(p))
    return p

'''raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
        'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool']'''

raw_list = ['Rao2014-GM12878-MboI-allreps-filtered.10kb.cool']
depth_ratio = [4,8,16,32,48,64]
# 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'

methods = ['ours_400', 'deephic_40', 'hicsr_40', 'low']
me_dict = {'deephic_40':'Deephic', 'hicsr_40':'HiCSR', 'ours_400':'EnHiC', 'low':'LR'}
labels = [me_dict[f] for f in methods]

chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']

metrics = ['GenomeDISCO', 'HiC-Spector', 'HiCRep']

data = list()
hic_ = raw_list[0]
c = cooler.Cooler(os.path.join('data', 'raw', hic_))
cl = hic_.split('-')
ctype = cl[1]+'_'+cl[2]

dict1 = dict(c.chromsizes)
chrid = sorted(dict1, key=dict1.get, reverse = False)
chrsize = {}
sorted_keys = sorted(dict1, key=dict1.get)
tmp = ['chr'+f for f in chromosomes]
for w in sorted_keys:
    if w in tmp:
        chrsize[w[3:]] = dict1[w]
print(chrsize)

hicrep_enhic = []
hicrep_hicsr = []

for dr in depth_ratio:
    if dr == 16:
        T = 'Y'
    else:
        T = 'N'
    ratio_name = '_'.join(cl[0:3]) +'-' + str(dr) + '_10kb'
    for chro in chromosomes:
        inpath = os.path.join('.', 'experiment', 'seq_depth_ratio', ratio_name, 
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
                        data.append([dr, T, me, chro, chrsize[chro], mc, l[i+2]])
                        if dr == 8 and me == 'ours_400' and mc == 'HiCRep':
                            hicrep_enhic.append(float(l[i+2]))
                        if dr == 8 and me == 'hicsr_40' and mc == 'HiCRep':
                            hicrep_hicsr.append(float(l[i+2]))
        fin.close()

s = pd.DataFrame(data, columns=["ratio", "base", "method", "chromosome", "chromosome length", "metric", "value"])

print(s)
print('enhic 8 hicrep', hicrep_enhic)
print('hicsr 8 hicrep', hicrep_hicsr)
stats.ttest_ind(hicrep_hicsr, hicrep_enhic, alternative='two-sided')
stats.ttest_ind(hicrep_hicsr, hicrep_enhic, alternative='greater')
# ttest_greater(hicrep_hicsr, hicrep_enhic)

output_dir = os.path.join('.', 'experiment', 'seq_depth_ratio')

sns.set(font_scale=2)
for mc in metrics:
    data = s.loc[s['metric']==mc]
    data = data.explode('value')
    data['value'] = data['value'].astype('float')
    
    fig, ax = plt.subplots()
    g = sns.catplot(ax = ax, x="ratio", y="value", hue="method", hue_order=methods, data=data, kind="point", capsize=.1, orient="v", height=6, aspect=2)
    # g = sns.lineplot(ax=ax, data=data, x="ratio", y="value", hue="method", style="base", markers=True)
    g.set_axis_labels("Downsampled ratio","Score")
    # g.set_xticklabels(rotation=30)
    if 'Genome' in mc:
        plt.ylim(-.2, 1.0)
    else:
        plt.ylim(0.7, 1.01)
    plt.gcf().subplots_adjust(bottom=0.15, top=0.92, left=0.06, right=0.85)
    plt.title('{} scores'.format(mc), size=30)

    # title
    legend_title = 'Method'
    g._legend.set_title(legend_title)
    # replace labels
    for t, l in zip(g._legend.texts, labels): t.set_text(l)

    output = os.path.join(output_dir, 'figure-seq_depth')
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'line_{}_scores.pdf'.format(mc))
    plt.savefig(output, format='pdf')

'''output_dir = os.path.join('.', 'experiment', 'seq_depth_ratio')
for mc in metrics:
    data = s.loc[s['metric']==mc]
    data = data.explode('value')
    data['value'] = data['value'].astype('float')
    
    fig, ax = plt.subplots(figsize=(12,8))
    # ax = sns.catplot(y="cell type", x="value", hue="method", data=data, kind="violin", orient="h", height=12, aspect=.8, width=0.8, scale="width", scale_hue=False)
    # g = sns.catplot(ax = ax, y="cell type", x="value", hue="method", hue_order=methods, data=data, kind="box", orient="h", height=12, aspect=.9)
    g = sns.lineplot(ax=ax, data=data, x="chromosome length", y="value", hue="ratio", style="base", markers=True)
    g.set_xticks(list(chrsize.values()))
    g.set_xticklabels(list(chrsize.keys()), fontsize=20)
    # ax.set(xlabel='cell type', ylabel='scores')
    g.set_xlabel('Chromosome',fontsize=20)
    g.set_ylabel('Score',fontsize=20);
    # g.set(xlabel='Chromosome', ylabel='Score')
    # g.set_axis_labels("Chromosome", "Score")
    plt.gcf().subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)
    if 'GenomeDISCO' in mc:
        plt.ylim(0.7, 0.9)
    else:
        plt.ylim(0.8, 1.0)
    plt.title('{} scores'.format(mc), size=24)

    # title
    legend_title = 'Method'
    g._legend.set_title(legend_title)
    # replace labels
    for t, l in zip(g._legend.texts, labels): t.set_text(l)

    output = os.path.join(output_dir, 'figure-seq_depth')
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'metrics_{}_scores.pdf'.format(mc))
    plt.savefig(output, format='pdf')'''
