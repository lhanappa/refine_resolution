import os
import numpy as np
from scipy.sparse import coo_matrix, triu

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

name = "/Volumes/GoogleDrive/My Drive/proj/refine_resolution/data/raw/Rao2014-HMEC-MboI-allreps-filtered.100kb.cool"
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

"""
# hicConvertFormat -m matrix.hic --inputFormat hic --outputFormat cool -o matrix.cool --resolutions 10000

script_work_dir = os.path.join(".", "demo")
matrix = os.path.join(".", "demo", "demo.cool")
outPrefix = "myHiCmatrix"
cmd = ["hicConvertFormat", 
        "-m", matrix, 
        "--inputFormat", "cool",
        "--outputFormat", "h5",
        "-o", "demo.h5"
    ]
process.append(subprocess.Popen(cmd, cwd=script_work_dir))
for p in process:
    p.wait()"""

script_work_dir = os.path.join(".", "demo")
matrix = os.path.join(".", "demo.cool")
outPrefix = "myHiCmatrix"
cmd = ["hicFindTADs", 
        "-m", matrix, 
        "--minDepth", "300000",
        "--maxDepth", "1000000",
        "--outPrefix", outPrefix,
        "--thresholdComparisons", "0.05",
        "--delta", "0.05",
        "--correctForMultipleTesting", "fdr",
        "--chromosomes", "chr10"
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
    p.wait()