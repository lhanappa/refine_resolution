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

def filter_diag_boundary(hic, diag_k=1, boundary_k=None):
    if boundary_k is None:
        boundary_k = hic.shape[0]-1
    boundary_k = min(hic.shape[0]-1, boundary_k)
    filter_m = np.tri(N=hic.shape[0], k=boundary_k)
    filter_m = np.triu(filter_m, k=diag_k)
    filter_m = filter_m + np.transpose(filter_m)
    return np.multiply(hic, filter_m)

def generate_fragments(chromosome, matrix, bins, output):
    # bins pandas dataframe, 
    chro_name = str(chromosome)
    hit_count = (matrix.sum(axis=0)).flatten()
    hit_count = hit_count.astype(int)
    mid_points = (bins[:, 1].flatten() + bins[:, 2])/2
    mid_points = mid_points.astype(int)
    with open(os.path.join(output+'_fragments.txt'), 'w+') as f:
        for i, mp in enumerate(mid_points):
            line = '{}\t0\t{}\t{}\t0\n'.format(chro_name, mp, hit_count[i])
            f.write(line)
    f.close()

    with open(os.path.join(output+'_fragments.txt'), 'rb') as f_in:
        with gzip.open(os.path.join(output+'_fragments.txt.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        f_out.close()
    f_in.close()
    os.remove(os.path.join(output+'_fragments.txt'))


def generate_interactions(chromosome, matrix, bins, output):
    chro_name = str(chromosome)
    mid_points = (bins[:, 1] + bins[:, 2])/2
    mid_points = mid_points.astype(int)
    mat = matrix.astype(int)
    coo_data = coo_matrix(mat)
    idx1 = mid_points[coo_data.row]
    idx2 = mid_points[coo_data.col]
    data = coo_data.data
    with open(os.path.join(output+'_interactions.txt'), 'w+') as f:
        for i, mp in enumerate(data):
            line = '{}\t{}\t{}\t{}\t{}\n'.format(chro_name, idx1[i], chro_name, idx2[i], data[i])
            f.write(line)
    f.close()

    with open(os.path.join(output+'_interactions.txt'), 'rb') as f_in:
        with gzip.open(os.path.join(output+'_interactions.txt.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        f_out.close()
    f_in.close()
    os.remove(os.path.join(output+'_interactions.txt'))

def geneate_biases_ICE(chromosome, matrix, bins, output):
    chro_name = str(chromosome)
    mid_points = (bins[:, 1] + bins[:, 2])/2
    mid_points = mid_points.astype(int)
    X, bias = normalization.ICE_normalization(matrix, output_bias=True)
    bias = bias.flatten().astype(float)
    with open(os.path.join(output+'_bias.txt'), 'w+') as f:
        for mp, bs in zip(mid_points, bias):
            line = '{}\t{}\t{}\n'.format(chro_name, mp, bs)
            f.write(line)
    f.close()
    with open(os.path.join(output+'_bias.txt'), 'rb') as f_in:
        with gzip.open(os.path.join(output+'_bias.txt.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        f_out.close()
    f_in.close()
    os.remove(os.path.join(output+'_bias.txt'))

def generate_fithic_files(cool_file, chromosome, start, end, output):
    hic = cooler.Cooler(cool_file)
    start = max(0, int(start))
    if end > hic.chromsizes['chr{}'.format(chromosome)]:
        length = end - start
        end = hic.chromsizes['chr{}'.format(chromosome)]
        start = end - length
    region = ('chr{}'.format(chromosome), start, end)
    hic_mat = hic.matrix(balance=True).fetch(region)
    hic_bins = hic.bins().fetch(region)
    weight = hic_bins['weight']
    idx = np.array(np.where(weight)).flatten()
    hic_bins = (hic_bins.to_numpy()).reshape((-1, 4))[idx, :]
    hic_mat = hic_mat[idx,:]
    hic_mat = hic_mat[:,idx]
    generate_fragments(chromosome, hic_mat, hic_bins, output)
    generate_interactions(chromosome, hic_mat, hic_bins, output)
    geneate_biases_ICE(chromosome, hic_mat, hic_bins, output)


def fithic_cmd(input_dir, prefix, resolution, low_dis, up_dis, start, end):
    # fithic -f high_chr22_fragments.txt.gz -i high_chr22_interactions.txt.gz -o ./ -r 10000 -t high_chr22_bias.txt.gz -L 0 -U 1000000 -v
    fragment = prefix+'_fragments.txt.gz'
    interaction = prefix+'_interactions.txt.gz'
    bias = prefix+'_bias.txt.gz'
    output = '{}_{}_{}'.format(prefix, start, end)

    cmd = ["fithic", 
            "-f", fragment, 
            "-i", interaction,
            "-o", output,
            '-r', str(resolution),
            "-t", bias,
            "-L", str(low_dis),
            "-U", str(up_dis),
            ]
    print('fithic cmd: {}'.format(' '.join(cmd)))
    return cmd


def extract_si(data, q_value_threshold=None):
    si = np.concatenate( (  data['fragmentMid1'].to_numpy().reshape((-1,1)), 
                            data['fragmentMid2'].to_numpy().reshape((-1,1)), 
                            data['q-value'].to_numpy().reshape((-1,1))
                            ), axis=1)
    diff = np.abs(si[:,0]-si[:,1]).reshape((-1,1))
    si = np.concatenate((si, diff), axis=1)
    if q_value_threshold is not None:
        idx = np.array(np.where(si[:,2]<q_value_threshold)).flatten()
        si = si[idx, :]
    si = si.reshape((-1,4))
    return si


def load_si(path, chromosome, model_name, resolution, low_dis, up_dis, start, end):
    path = os.path.join(path, 'output_{}_{}'.format(start, end))
    prefix = '{}_chr{}_{}_{}'.format(model_name, chromosome, start, end)
    model_path = os.path.join(path, prefix, 'FitHiC.spline_pass1.res10000.significances.txt.gz')
    si = dict()
    if not os.path.isfile( model_path):
        return si
    model_data = pd.read_csv(model_path, compression='gzip', header=0, sep='\t')
    if model_data.empty:
        return si

    q_value = 0.05
    model_si = extract_si(model_data, q_value_threshold=q_value)
    keys = np.unique(model_si[:,3])
    for k in keys:
        idx = np.array(np.where(model_si[:,3]==k)).flatten()
        si[k] = np.unique(model_si[idx,0].flatten())
    return si


def merge_si(d0, d1):
    key0 = list(d0.keys())
    key1 = list(d1.keys())
    keys = np.union1d(key0, key1)
    si = dict()
    for k in keys:
        if k not in key0:
            si[k] = np.unique(d1[k])
        elif k not in key1:
            si[k] = np.unique(d0[k])
        else:
            si[k] = np.union1d(d0[k], d1[k])
    return si

def jaccard_score(models, ground):
    dis1 = list(ground.keys())
    model_js = dict()
    for m, v in models.items():
        js_array = []
        dis0 = list(v.keys())
        dis = np.intersect1d(dis0, dis1)
        for d in ground.keys():
            if d in v.keys():
                HR_set = np.unique(ground[d])
                model_set = np.unique(v[d])
                intersection = len(np.intersect1d(HR_set, model_set))
                union = len(np.union1d(HR_set, model_set))
                if union != 0:
                    js = intersection/union
                    js_array.append([d/resolution, js])
            else:
                js_array.append([d/resolution, 0])
        js_array = np.array(js_array).reshape((-1,2))
        model_js[m] = js_array
    return model_js


def plot_demo(source_dir, chromosome, model_name, resolution, low_dis, up_dis, start, end):
    cool_file = os.path.join(source_dir, '{}_chr{}.cool'.format(model_name, chromosome))
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
    output = os.path.join(source_dir, 'figure', '{}_{}'.format(start, end))
    os.makedirs(output, exist_ok=True)
    plt.savefig(os.path.join(output, 'demo_{}.pdf'.format(legend[name])), format='pdf')
    plt.savefig(os.path.join(output, 'demo_{}.jpg'.format(legend[name])), format='jpg')



"""chromsizes = {
'chr1':     249250621,
'chr2':     243199373,
'chr3':     198022430,
'chr4':     191154276,
'chr5':     180915260,
'chr6':     171115067,
'chr7':     159138663,
'chr8':     146364022,
'chr9':     141213431,
'chr10':    135534747,
'chr11':    135006516,
'chr12':    133851895,
'chr13':    115169878,
'chr14':    107349540,
'chr15':    102531392,
'chr16':     90354753,
'chr17':     81195210,
'chr18':     78077248,
'chr19':     59128983,
'chr20':     63025520,
'chr21':     48129895,
'chr22':     51304566,
'chrX':     155270560,
'chrY':      59373566,
'chrM':         16571}"""

if __name__ == '__main__':
    # methods = ['ours_400', 'hicsr_40', 'deephic_40', 'high', 'low']

    # cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
    hic_info = cooler.Cooler(os.path.join('.', 'data', 'raw', cool_file))
    resolution = int(hic_info.binsize) # 10000, 10kb

    # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    chromosomes = [str(sys.argv[1])] # ['17', '18', '19', '20', '21', '22', 'X']# [str(sys.argv[1])]
    [start, end] = [int(sys.argv[2]), int(sys.argv[3])]
    genome_dis = int(100)
    window_len = int(200)
    [low, up] = np.array([0, genome_dis], dtype=int)*resolution

    chrom_js = dict()
    for chro in chromosomes:
        path = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro))
        files = [f for f in os.listdir(path) if '.cool' in f]
        hic_chrom_len = np.ceil(hic_info.chromsizes['chr{}'.format(chro)]/resolution)

        model_all_si = dict()
        hr_all_si = dict()

        queue = []
        print(start, end)
        source_dir = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro))
        for file in files:
            m = file.split('_')[0:-1]
            m = '_'.join(m)

            # plot_significant_interactions(source_dir, chro, m, resolution, low_dis=low, up_dis=up, start=start, end=end)
            p = Process(target=plot_demo, args=(source_dir, chro, m, resolution, low, up, start, end))
            queue.append(p)
            p.start()

        for p in queue:
            p.join()

