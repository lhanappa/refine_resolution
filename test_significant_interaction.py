import os, sys, shutil, gzip
import numpy as np
from scipy.sparse import coo_matrix, triu
from scipy.spatial import distance
import subprocess
import pandas as pd

import cooler
from iced import filter
from iced import normalization

from matplotlib import pyplot as plt

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
    end = min(hic.chromsizes['chr{}'.format(chromosome)], int(end))
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

def extract_si(data):
    si = np.concatenate( (  data['fragmentMid1'].to_numpy().reshape((-1,1)), 
                            data['fragmentMid2'].to_numpy().reshape((-1,1)), 
                            data['q-value'].to_numpy().reshape((-1,1))
                            ), axis=1)
    diff = np.abs(si[:,0]-si[:,1]).reshape((-1,1))
    si = np.concatenate((si, diff), axis=1)
    return si

def jaccard_score_with_HR(path, chromosome, model_name, resolution, low_dis, up_dis, start, end):
    path = os.path.join(path, 'output_{}_{}'.format(start, end))
    prefix = 'high_chr{}_{}_{}'.format(chromosome, start, end)
    HR_path = os.path.join(path, prefix, 'FitHiC.spline_pass1.res10000.significances.txt.gz')
    prefix = '{}_chr{}_{}_{}'.format(model_name, chromosome, start, end)
    model_path = os.path.join(path, prefix, 'FitHiC.spline_pass1.res10000.significances.txt.gz')

    HR_data = pd.read_csv(HR_path, compression='gzip', header=0, sep='\t')
    model_data = pd.read_csv(model_path, compression='gzip', header=0, sep='\t')

    HR_si = extract_si(HR_data)
    model_si = extract_si(model_data)

    idx = np.array(np.where(HR_si[:,2]<0.05)).flatten()
    HR_si = HR_si[idx, :]

    idx = np.array(np.where(model_si[:,2]<0.05)).flatten()
    model_si = model_si[idx, :]

    js_array = []
    for dis in np.arange(low_dis, up_dis+1, resolution):
        HR_idx = np.array(np.where(HR_si[:,3]==dis)).flatten()
        HR_set = np.unique(HR_si[HR_idx, 0].flatten())

        model_idx = np.array(np.where(model_si[:,3]==dis)).flatten()
        model_set = np.unique(model_si[model_idx, 0].flatten())

        intersection = len(np.intersect1d(HR_set, model_set))
        union = len(np.union1d(HR_set, model_set))
        if union != 0:
            js = intersection/union
            if js > 0:
                js_array.append([dis/resolution, js])
    js_array = np.array(js_array)
    print(model_name)
    print(js_array)
    return js_array, HR_data, model_data


def plot_significant_interactions(source_dir, chromosome, model_name, resolution, low_dis, up_dis, start, end):
    start = int(start)
    end = int(end)
    cool_file = os.path.join(source_dir, '{}_chr{}.cool'.format(model_name, chromosome))
    hic = cooler.Cooler(cool_file)
    region = ('chr{}'.format(chromosome), start, end)
    hic_mat = hic.matrix(balance=True).fetch(region)
    hic_mat = normalization.ICE_normalization(hic_mat)
    # hic_bins = hic.bins().fetch(region)
    # weight = hic_bins['weight']
    # filter_idx = np.array(np.where(weight==1)).flatten()
    

    prefix = '{}_chr{}_{}_{}'.format(model_name, chromosome, start, end)
    model_path = os.path.join(source_dir, 'output_{}_{}'.format(start, end), prefix, 'FitHiC.spline_pass1.res10000.significances.txt.gz')
    model_data = pd.read_csv(model_path, compression='gzip', header=0, sep='\t')
    model_si = extract_si(model_data)

    q_idx = np.array(np.where(model_si[:, 2]<0.05)).flatten()
    si_x = np.floor((model_si[q_idx,0].flatten() - start)/resolution)
    si_y = np.floor((model_si[q_idx,1].flatten() - start)/resolution)

    fig, ax0 = plt.subplots()
    cmap = plt.get_cmap('coolwarm')
    X, Y = np.meshgrid(np.arange(hic_mat.shape[0]), np.arange(hic_mat.shape[1]))
    # X, Y = np.meshgrid(np.arange(len(filter_idx)), np.arange(len(filter_idx)))
    hic_mat = filter_diag_boundary(hic_mat, diag_k=1, boundary_k=200)
    Z = np.log1p(hic_mat)
    # Z = Z[filter_idx,:]
    # Z = Z[:,filter_idx]
    im = ax0.pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=8)
    fig.colorbar(im, ax=ax0)
    ax0.scatter(si_x.flatten(), si_y.flatten(), color="gold", s=.1)
    ax0.set_title('{} log1p Heatmap'.format(model_name))

    fig.tight_layout()
    output = os.path.join(source_dir, 'figure', '{}_{}'.format(start, end))
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, '{}_chr{}_{}_{}.pdf'.format(model_name, chromosome, start, end))
    plt.savefig(output, format='pdf')


def plot_jaccard_score(source_dir, model_js):
    fig, ax0 = plt.subplots()
    for key, value in model_js.items():
        x = value[:,0]
        y = value[:,1]
        ax0.plot(x, y, label='{}'.format(key))
    fig.tight_layout()
    output = os.path.join(source_dir, 'figure', '{}_{}'.format(start, end))
    os.makedirs(output, exist_ok=True)
    output = os.path.join(output, 'jaccard_scores_{}_{}.pdf'.format(start, end))
    plt.savefig(output, format='pdf')


if __name__ == '__main__':
    # methods = ['ours_400', 'hicsr_40', 'deephic_40', 'high', 'low']

    # cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]

    # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    chromosomes = [str(sys.argv[1])]
    resolution = 10000
    [low, up] = np.array([0, 100], dtype=int)*resolution
    for chro in chromosomes:
        path = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro))
        files = [f for f in os.listdir(path) if '.cool' in f]

        [start, end] = np.array([2200, 2500], dtype=int)*resolution
        """process = []
        for file in files:
            m = file.split('.')[0]
            source = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro), file)
            dest =  os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro), 'output_{}_{}'.format(start, end))
            os.makedirs(dest, exist_ok=True)
            generate_fithic_files(source, chro, start, end, output=os.path.join(dest, m))
            cmd = fithic_cmd(input_dir=dest, prefix=m, resolution=resolution, low_dis=low, up_dis=up, start=start, end=end)
            script_work_dir = dest
            process.append(subprocess.Popen(cmd, cwd=script_work_dir))
        for p in process:
            p.wait()"""

        model_js = dict()
        for file in files:
            m = file.split('_')[0:-1]
            m = '_'.join(m)
            source_dir = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro))
            # plot_significant_interactions(source_dir, chro, m, resolution, low_dis=low, up_dis=up, start=start, end=end)

            if 'high' not in m:
                js, _, _ = jaccard_score_with_HR(source_dir, chro, m, resolution, low_dis=low, up_dis=up, start=start, end=end)
                model_js[m] = js
            plot_jaccard_score(source_dir, model_js)


