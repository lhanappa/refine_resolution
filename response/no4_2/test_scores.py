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
        file = 'reproducibility.chr{}.txt'.format(chrom)
        inpath = os.path.join(path, file)
        print(inpath)
        if not os.path.exists(inpath):
            continue
        with open(inpath, 'r') as fin:
            for line in fin:
                l = line.split()
                print(l)
                if '#' in l[0]:
                    continue
                for i,m in enumerate(methods):
                    if m == l[0]:
                        me1 = i
                    if m == l[1]:
                        me2 = i
                disco_data[me1, me2] = disco_data[me1, me2] + float(l[2])
                hicrep_data[me1, me2] = hicrep_data[me1, me2] + float(l[3])
                cnt[me1, me2] = cnt[me1, me2]+1
                cnt[me2, me1] = cnt[me1, me2]
                hicrep_data[me2, me1] = hicrep_data[me1, me2]
                disco_data[me2, me1] = disco_data[me1, me2]
        fin.close()

    mean = hicrep_data/cnt
    print(hicrep_data)
    print(cnt)
    print(mean)
    mask = np.zeros_like(mean)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        g = sns.heatmap(ax=ax, data=mean, mask=mask, vmin=0.9, annot=True, square=True)
        g.set(xticklabels=methods)
        g.set(yticklabels=methods)
    output_dir = os.path.join('.')
    output = os.path.join(output_dir, 'figure')
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'hicrep_scores.pdf')
    plt.savefig(output, format='pdf')


if __name__ == '__main__':
    # methods = ['deephic_40', 'hicsr_40', 'ours_80', 'ours_200', 'ours_400', 'high', 'low']
    methods = ['rep1', 'rep2', 'rep3', 'rep4', 'multiple']
    run(methods)