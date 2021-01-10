import os, sys, shutil, gzip
import numpy as np
from scipy.sparse import coo_matrix, triu
from scipy.spatial import distance
import subprocess
import pandas as pd

import cooler
from iced import filter
from iced import normalization

# using fithic to find significant interactions by CLI


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

def geneate_biases_ICE(chromosome, matrix, bins, output):
    chro_name = str(chromosome)
    mid_points = (bins[:, 1] + bins[:, 2])/2
    mid_points = mid_points.astype(int)
    X, bias = ICE_normalization(matrix, output_bias=True)
    bias = bias.flatten().astype(float)
    with open(os.path.join(output+'_bias.txt'), 'w+') as f:
        for mp, bs in zip(mid_points, bias):
            line = '{}\t{}\t{}\n'.format(chro_name, mp, bs)
            f.write(line)
    f.close()
    with open(os.path.join(output+'_bias.txt'), 'rb') as f_in:
        with gzip.open(os.path.join(output+'_bias.txt.gz'), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

def generate_fithic_files(cool_file, chromosome, start, end, output):
    start = int(start)
    end = int(end)
    hic = cooler.Cooler(cool_file)
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

def fit_significant_interaction(input_dir, prefix, resolution, low_dis, up_dis):
    # fithic -f high_chr22_fragments.txt.gz -i high_chr22_interactions.txt.gz -o ./ -r 10000 -t high_chr22_bias.txt.gz -L 0 -U 1000000 -v
    fragment = prefix+'_fragments.txt.gz'
    interaction = prefix+'_interactions.txt.gz'
    bias = input_dir, prefix+'_bias.txt.gz'
    output = prefix

    script_work_dir = input_dir
    cmd = ["fithic", 
            "-f", fragment, 
            "-i", interaction,
            "-o", output,
            '-r', resolution,
            "-t", bias,
            "-L", low_dis,
            "-U", up_dis,
            ]
    process.append(subprocess.Popen(cmd, cwd=script_work_dir))
    return process


if __name__ == '__main__':
    # methods = ['ours_400', 'hicsr_40', 'deephic_40', 'high', 'low']

    # cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
    destination_path = os.path.join('.','experiment', 'significant_interactions', cell_type)

    # chromosomes = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X']
    # chromosomes = [ '22' ]
    chromosomes = [str(sys.argv[1])]
    resolution = 10000
    [start, end] = np.array([2200, 2500], dtype=int)*resolution
    [low, up] = np.array([0, 100], dtype=int)*resolution
    for chro in chromosomes:
        path = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro))
        files = [f for f in os.listdir(path) if '.cool' in f]
        for file in files:
            m = file.split('.')[0]
            source = os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro), file)
            dest =  os.path.join('.', 'experiment', 'significant_interactions', cell_type, 'chr{}'.format(chro), 'output')
            os.makedirs(dest, exist_ok=True)
            dest = os.path.join(dest, m)
            generate_fithic_files(source, chro, start, end, output=dest)
            process = fit_significant_interaction(input_dir=dest, prefix=m, resolution=resolution, low_dis=low, up_dis=up)
        for p in process:
            p.wait()
    
