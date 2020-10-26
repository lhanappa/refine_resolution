# this file qualify hic 1 and hic 2 by calling 3Dchromatin_replicateQC
# https://github.com/kundajelab/3DChromatin_ReplicateQC
# in Python 2.7
import numpy as np
import gzip
import os, io
import subprocess

def merge_contact(files = [], method='', outdir='./'):
    outfile = '{}.txt.gz'.format(method)
    with gzip.open(os.path.join(outdir, outfile), 'w') as fout:
        for file in files:
            with gzip.open(file, 'rb') as gz:
                f = io.BufferedReader(gz)
                flines = f.readlines()
                fout.writelines(flines)
            gz.close()
    fout.close()


def merge_bins(file, outdir='./', outfile = 'Bins.bed.gz'):
    with gzip.open(os.path.join(outdir, outfile), 'w') as fout:
        with gzip.open(file, 'rb') as gz:
            f = io.BufferedReader(gz)
            for line in f.readline():
                l = line
                fout.write(l)
        gz.close()
    fout.close()

def generate_paramters():
    pass

def generate_metadata_samples(samples):
    pass

def generate_pairs(file_list1, file_list2):
    pass

def get_high_low_resolution():
    pass

"""
3DChromatin_ReplicateQC preprocess --running_mode sge --metadata_samples examples/metadata.samples --bins examples/Bins.w50000.bed.gz --outdir examples/output --parameters_file examples/example_parameters.txt
3DChromatin_ReplicateQC qc --running_mode sge --metadata_samples examples/metadata.samples --outdir examples/output --methods QuASAR-QC
3DChromatin_ReplicateQC concordance --running_mode sge --metadata_pairs examples/metadata.pairs --outdir examples/output --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep
3DChromatin_ReplicateQC summary --running_mode sge --metadata_samples examples/metadata.samples --metadata_pairs examples/metadata.pairs --bins examples/Bins.w50000.bed.gz --outdir examples/output --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep,QuASAR-QC
3DChromatin_ReplicateQC cleanup --running_mode sge --outdir examples/output
"""

def run():
    methods = ['hicgan', 'deephic', 'hicsr', 'ours', 'high-resolution', 'low-resolution']
    bin_file = ''
    output_dir = ''
    merge_bins(bin_file, output_dir)
    samples = dict()
    for i, m in enumerate(methods):
        contact_files = []
        [path, file] = merge_contact(contact_files, m, output_dir)
        samples[m] = os.path.join(path, file)

    generate_paramters()
    generate_metadata_samples(samples)

# 3DChromatin_ReplicateQC preprocess 
# --running_mode sge 
# --metadata_samples examples/metadata.samples 
# --bins examples/Bins.w50000.bed.gz 
# --outdir examples/output 
# --parameters_file examples/example_parameters.txt
    script_work_dir = './'
    cmd = ["3DChromatin_ReplicateQC", "preprocess", 
        "--running_mode", data_fp, 
        "--metadata_samples", output_path, 
        "--bins", "HiCSR", 
        "--outdir", model_fp,
        "--parameters_file", str(resolution)]
    process = subprocess.run(cmd, cwd=script_work_dir)

# 3DChromatin_ReplicateQC concordance 
# --running_mode sge 
# --metadata_pairs examples/metadata.pairs 
# --outdir examples/output 
# --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep
    script_work_dir = './'
    cmd = ["3DChromatin_ReplicateQC", "preprocess", 
        "--running_mode", data_fp, 
        "--metadata_samples", output_path, 
        "--bins", "HiCSR", 
        "--outdir", model_fp,
        "--parameters_file", str(resolution)]
    process = subprocess.run(cmd, cwd=script_work_dir)

# 3DChromatin_ReplicateQC summary 
# --running_mode sge 
# --metadata_samples examples/metadata.samples 
# --metadata_pairs examples/metadata.pairs 
# --bins examples/Bins.w50000.bed.gz 
# --outdir examples/output 
# --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep    script_work_dir = './'
    cmd = ["3DChromatin_ReplicateQC", "preprocess", 
        "--running_mode", data_fp, 
        "--metadata_samples", output_path, 
        "--bins", "HiCSR", 
        "--outdir", model_fp,
        "--parameters_file", str(resolution)]
    process = subprocess.run(cmd, cwd=script_work_dir)
