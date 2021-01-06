import os, sys
import numpy as np
from scipy.sparse import coo_matrix, triu
from scipy.spatial import distance
import subprocess
import pandas as pd

import cooler
from iced import filter
from iced import normalization
# using hicexplorer to find tad boundary by CLI

# $ hicFindTADs -m myHiCmatrix.h5 \
# --outPrefix myHiCmatrix_min3000_max31500_step1500_thres0.05_delta0.01_fdr \
# --minDepth 3000 \
# --maxDepth 31500 \
# --step 1500 \
# --thresholdComparisons 0.05 \
# --delta 0.01 \
# --correctForMultipleTesting fdr \
# -p 64

"""name = "/Volumes/GoogleDrive/My Drive/proj/refine_resolution/data/raw/Rao2014-HMEC-MboI-allreps-filtered.100kb.cool"
c = cooler.Cooler(name)
resolution = c.binsize
mat = c.matrix(balance=True).fetch('chr10')
mat = filter.filter_low_counts(mat, percentage=0.01)
mat = normalization.ICE_normalization(mat)

bins = c.bins().fetch('chr10')

print(mat.shape)
print(resolution)

b = {'chrom': ['chr10']*mat.shape[0], 'start': resolution*np.arange(mat.shape[0]), 'end': resolution*np.arange(1, 1+mat.shape[0]), 'weight': [1.0]*mat.shape[0]}
bins = pd.DataFrame(data = b)
print(bins)

mat = triu(mat, format='coo')
p = {'bin1_id': mat.row, 'bin2_id': mat.col, 'count': mat.data}
pixels = pd.DataFrame(data = p)

uri = os.path.join(".", "demo", "demo.cool")
cooler.create_cooler(cool_uri=uri, bins=bins, pixels=pixels)

script_work_dir = os.path.join(".", "demo")
matrix = os.path.join(".", "demo.cool")
outPrefix = os.path.join(".", "output", "myHiCmatrix")
cmd = ["hicFindTADs", 
        "-m", matrix, 
        "--minDepth", "300000",
        "--maxDepth", "1000000",
        "--outPrefix", outPrefix,
        "--thresholdComparisons", "0.05",
        "--delta", "0.05",
        "--correctForMultipleTesting", "fdr",
    ]
process = list()
process.append(subprocess.Popen(cmd, cwd=script_work_dir))
for p in process:
    p.wait()

script_work_dir = os.path.join(".", "demo")
matrix = os.path.join(".", "demo.cool")
outfile = "myHiCmatrix"
cmd = ["hicPlotMatrix", 
        "--log1p", 
        "--matrix", matrix, 
        "--outFileName", outfile,
    ]
process = list()
process.append(subprocess.Popen(cmd, cwd=script_work_dir))
for p in process:
    p.wait()"""

def load_bedfile(file):
    starts = []
    ends = []
    with open(file, 'r') as f:
        for line in f:
            l = line.split()
            starts.append(int(l[1]))
            ends.append(int(l[2]))
    f.close()
    return [np.array(starts).reshape(-1,1), np.array(ends).reshape(-1,1)]

def identify(data_1, data_2, shift=0):
    a_starts, a_ends = data_1
    b_starts, b_ends = data_2

    dis_starts = distance.cdist(a_starts, b_starts)
    dis_ends = distance.cdist(a_ends, b_ends)

    mask_starts = (dis_starts <= shift)
    mask_ends = (dis_ends <= shift)
    mask = np.logical_and(mask_starts, mask_ends)
    intersectionx, intersectiony = np.where(mask)
    intersection = min(len(intersectionx), len(intersectiony))
    seta = mask.shape[0]
    setb = mask.shape[1]

    only_a = seta - intersection
    only_b = setb - intersection
    return [intersection, only_a, only_b, mask]

def check_tad_boundary(input_path, chromosomes, models_1, models_2=['high'], shift = 0):
    for chro in chromosomes:
        script_work_dir = os.path.join(input_path, 'chr{}'.format(chro), 'output')
        output = script_work_dir
        for m1 in models_1:
            for m2 in models_2:
                filename_1 = '{}_chr{}_domains.bed'.format(m1, chro)
                filename_2 = '{}_chr{}_domains.bed'.format(m2, chro)
                in1 = os.path.join(script_work_dir, filename_1)
                in2 = os.path.join(script_work_dir, filename_2)

                if not os.path.isfile(in1):
                    estimate_tad_boundary([chro], [m1], input_path=input_path)
                if not os.path.isfile(in2):
                    estimate_tad_boundary([chro], [m2], input_path=input_path)

                data_1 = load_bedfile(in1)
                data_2 = load_bedfile(in2)
                [intersection, only_a, only_b, mask] = identify(data_1, data_2, shift=shift)
                jaccard_score = float(intersection)/float(intersection+only_a+only_b)
                print('Jaccard score: {}, intersection: {}, {} only in {}, {} only in {}'.format(jaccard_score, intersection, only_a, m1, only_b, m2))
                with open(os.path.join(script_work_dir, 'TAD_Jaccard_score.txt'), 'a+') as f:
                    line = 'chr{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(chro, m1, m2, jaccard_score, intersection, only_a, only_b)
                    f.write(line)
                f.close()

def estimate_tad_boundary(chromosomes, models, input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    for chro in chromosomes:
        process = list()
        for m in models:
            script_work_dir = os.path.join(input_path, 'chr{}'.format(chro))
            filename = '{}_chr{}'.format(m, chro)
            infile = os.path.join('{}.cool'.format(filename))
            out = os.path.join(output_path, 'chr{}'.format(chro), 'output')
            os.makedirs(out, exist_ok=True)
            out = os.path.join('output', filename)
            cmd = ["hicFindTADs", 
                    "-m", infile, 
                    "--minDepth", "60000",
                    "--maxDepth", "200000",
                    "--step", "10000",
                    "--outPrefix", out,
                    "--thresholdComparisons", "0.01",
                    "--delta", "0.01",
                    "--correctForMultipleTesting", "fdr",
                ]
            process.append(subprocess.Popen(cmd, cwd=script_work_dir))
        for p in process:
            p.wait()

"""def diff_tad_boundary(chromosomes, models, input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    for chro in chromosomes:
        process = list()
        for m in models:
            script_work_dir = os.path.join(input_path, 'chr{}'.format(chro))
            filename = '{}_chr{}'.format(m, chro)
            infile = os.path.join('{}.cool'.format(filename))
            out = os.path.join(output_path, 'chr{}'.format(chro), 'output')
            os.makedirs(out, exist_ok=True)
            out = os.path.join('output', filename)
            cmd = ["hicFindTADs", 
                    "-m", infile, 
                    "--minDepth", "60000",
                    "--maxDepth", "200000",
                    "--step", "10000",
                    "--outPrefix", out,
                    "--thresholdComparisons", "0.01",
                    "--delta", "0.01",
                    "--correctForMultipleTesting", "fdr",
                ]
            process.append(subprocess.Popen(cmd, cwd=script_work_dir))
        for p in process:
            p.wait()"""

def plot_hic(chromosomes, models, input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    for chro in chromosomes:
        process = list()
        for m in models:
            script_work_dir = os.path.join(input_path, 'chr{}'.format(chro))
            filename = '{}_chr{}'.format(m, chro)
            infile = os.path.join('{}.cool'.format(filename))
            out = os.path.join(output_path, 'chr{}'.format(chro), 'output')
            os.makedirs(out, exist_ok=True)
            out = os.path.join('output', filename)
            cmd = ["hicPlotMatrix", 
                        "--log1p", 
                        "--matrix", infile, 
                        "--outFileName", out,
                    ]
            process.append(subprocess.Popen(cmd, cwd=script_work_dir))
        for p in process:
            p.wait()

"""def plot_tad_boundary(chromosomes, models, input_path, output_path=None):
    if output_path is None:
        output_path = input_path

    for chro in chromosomes:
        process = list()
        for m in models:
            script_work_dir = os.path.join(input_path, 'chr{}'.format(chro))
            filename = '{}_chr{}'.format(m, chro)
            infile = os.path.join('{}.cool'.format(filename))
            out = os.path.join(output_path, 'chr{}'.format(chro), 'output')
            os.makedirs(out, exist_ok=True)
            out = os.path.join('output', filename)
            cmd = ["hicPlotTAD", 
                        "--log1p", 
                        "--matrix", infile, 
                        "--outFileName", out,
                    ]
            process.append(subprocess.Popen(cmd, cwd=script_work_dir))
        for p in process:
            p.wait()"""

"""def plot_tad_boundary(chromosomes, models, input_path):
    script_work_dir = os.path.join(".", "demo")
    outfile = "myHiCTADs"
    cmd = ["hicPlotTADs", 
            "--tracks", "tracks.ini", 
            "--region", "chr10:1500000-40000000",
            "--outFileName", outfile
        ]
    process = list()
    process.append(subprocess.Popen(cmd, cwd=script_work_dir))
    for p in process:
        p.wait()

    script_work_dir = os.path.join(".", "demo")
    outfile = "myHiCTADs"
    cmd = ["hicPlotTADs", 
            "--tracks", "tracks.ini", 
            "--region", "chr10:1500000-40000000",
            "--outFileName", outfile
        ]
    process = list()
    process.append(subprocess.Popen(cmd, cwd=script_work_dir))
    for p in process:
        p.wait()"""

if __name__ == '__main__':
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    # cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
    input_path = os.path.join('./experiment', 'tad_boundary', cell_type)

    chromosomes = [str(sys.argv[1])]
    # chromosomes = ['22', '21', '20', '19', 'X']
    # chromosomes = ['22']
    models = ['deephic_40', 'hicsr_40', 'ours_400', 'low'] # 'hicgan', 'ours_80', 'ours_200', 
    resolution = 10000
    estimate_tad_boundary(chromosomes, models, input_path=input_path)
    # plot_hic(chromosomes, models, input_path=input_path)
    check_tad_boundary(input_path=input_path, chromosomes=chromosomes, models_1=models, models_2=['high'], shift=resolution*3)
