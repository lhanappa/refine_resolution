import numpy as np
import os, sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run(methods):
    chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    metrics = ['GenomeDISCO', 'HiCRep']
    hicrep_data = np.zeros((len(methods), len(methods)))
    disco_data = np.zeros((len(methods), len(methods)))
    cnt = np.zeros((len(methods), len(methods)))
    for chrom in chromosomes:
        path = os.path.join('.', 'data', 'chr{}'.format(chrom), 'chromatin_qc', 'scores')
        file = 'reporducibility.chr{}.txt'.format(chrom)
        inpath = os.path.join(path, file)
        if not os.path.exists(inpath):
            continue
        with open(inpath, 'r') as fin:
            for line in fin:
                l = line.split()
                me1 = np.where(l[0]==methods)[0]
                me2 = np.where(l[1]==methods)[0]
                disco_data[me1, me2] = l[2]
                hicrep_data[me1, me2] = l[3]
                cnt[me1, me2] = cnt[me1, me2]+1
                cnt[me2, me1] = cnt[me1, me2]
        fin.close()

    mean = hicrep_data/cnt
    print(hicrep_data)
    print(cnt)
    print(mean)
    mask = np.zeros_like(mean)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(mean, mask=mask, vmax=.3, square=True)
    output_dir = os.path.join('.')
    output = os.path.join(output_dir, 'figure')
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'hicrep_scores.pdf')
    plt.savefig(output, format='pdf')


if __name__ == '__main__':
    # methods = ['deephic_40', 'hicsr_40', 'ours_80', 'ours_200', 'ours_400', 'high', 'low']
    methods = ['rep1', 'rep2', 'rep3', 'rep4', 'multiple']
    run(methods)