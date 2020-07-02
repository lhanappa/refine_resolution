"""
call 3DChromatin_ReplicateQC to qualify Hi-C matrix


3DChromatin_ReplicateQC run_all 
--metadata_samples examples/metadata.samples 
--metadata_pairs examples/metadata.pairs 
--bins examples/Bins.w50000.bed.gz 
--outdir examples/output 
*   This package need to switch env to 
    3dchromatin_replicate_qc
"""


from .operations import *
import numpy as np
import sys
import io
import os
import subprocess


def run_hicrep(script,
               f1,
               f2,
               bedfile,
               output_path='./',
               maxdist=int(2000000),
               resolution=int(10000),
               h=int(20),
               m1name='m1',
               m2name='m2'):

    cmd = ['Rscript --vanilla', script, f1, f2, output_path, str(
        maxdist), str(resolution), bedfile, str(h), m1name, m2name]
    print(' '.join(cmd))
    os.system(' '.join(cmd))
    #proc = subprocess.call([str(' '.join(cmd))],stdout=subprocess.PIPE)
    #stdout_value = proc.wait()
