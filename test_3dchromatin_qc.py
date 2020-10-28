# this file qualify hic 1 and hic 2 by calling 3Dchromatin_replicateQC
# https://github.com/kundajelab/3DChromatin_ReplicateQC
# in Python 2.7
import numpy as np
import gzip
import os, io
import subprocess


parameters = '''GenomeDISCO|subsampling	lowest
GenomeDISCO|tmin	3
GenomeDISCO|tmax	3
GenomeDISCO|norm	uniform
GenomeDISCO|scoresByStep	yes
GenomeDISCO|removeDiag	yes
GenomeDISCO|transition	yes
HiCRep|h	20
HiCRep|maxdist	2000000
HiC-Spector|n	20
QuASAR|rebinning	resolution
'''

def generate_parameters(chromosome, path='./experiment/evaluation'):
    path = os.path.join(path, 'chr{}'.format(chromosome))
    fout = open(os.path.join(path, 'qc_parameters.txt'), 'w+')
    fout.write(parameters)
    fout.close()

def generate_metadata_samples(methods, chromosome, path='./experiment/evaluation'):
    path = os.path.join(path, 'chr{}'.format(chromosome))
    files = [f for  f in os.listdir(path) if 'contact.gz' in f]
    fin = open(os.path.join(path, 'metadata_samples.txt'), 'w+')
    for file in files:
        f = os.path.join(path, file)
        absolute_path = os.path.abspath(f)
        method = file.split('_')[0]
        if method in methods:
            line = '{}\t{}\n'.format(method, absolute_path)
            fin.write(line)
    fin.close()


def generate_pairs(file_list1, file_list2, chromosome, path='./experiment/evaluation'):
    path = os.path.join(path, 'chr{}'.format(chromosome))
    with open(os.path.join(path, 'metadata_pairs.txt'), 'w+') as fin:
        for f1 in file_list1:
            for f2 in file_list2:
                l = [f1, f2]
                line = '\t'.join(l)+ '\n'
                fin.write(line)
    fin.close()

"""
3DChromatin_ReplicateQC preprocess --running_mode sge --metadata_samples examples/metadata.samples --bins examples/Bins.w50000.bed.gz --outdir examples/output --parameters_file examples/example_parameters.txt
3DChromatin_ReplicateQC qc --running_mode sge --metadata_samples examples/metadata.samples --outdir examples/output --methods QuASAR-QC
3DChromatin_ReplicateQC concordance --running_mode sge --metadata_pairs examples/metadata.pairs --outdir examples/output --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep
3DChromatin_ReplicateQC summary --running_mode sge --metadata_samples examples/metadata.samples --metadata_pairs examples/metadata.pairs --bins examples/Bins.w50000.bed.gz --outdir examples/output --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep,QuASAR-QC
3DChromatin_ReplicateQC cleanup --running_mode sge --outdir examples/output
"""

def run(
    methods = ['hicgan', 'deephic', 'hicsr', 'ours', 'high', 'low'],
    list1 = ['high'],
    list2 = ['hicgan', 'deephic', 'hicsr', 'ours', 'low'],
    chromosomes = ['22', '21', '20', '19', 'X']):
    print(chromosomes)
    for chro in chromosomes:
        generate_parameters(chro)
        generate_metadata_samples(methods, chro)
        generate_pairs(list1, list2, chro)

        # 3DChromatin_ReplicateQC preprocess 
        # --running_mode sge 
        # --metadata_samples examples/metadata.samples 
        # --bins examples/Bins.w50000.bed.gz 
        # --outdir examples/output 
        # --parameters_file examples/example_parameters.txt
        script_work_dir = './experiment/evaluation/chr{}'.format(chro)
        cmd = ["3DChromatin_ReplicateQC", "preprocess", 
            "--metadata_samples",  'metadata_samples.txt', 
            "--bins", 'bins_chr{}.bed.gz'.format(chro), 
            "--outdir", './output',
            "--methods", "GenomeDISCO,HiCRep,HiC-Spector",
            "--parameters_file", './qc_parameters.txt']
        process = subprocess.Popen(cmd, cwd=script_work_dir)

        # 3DChromatin_ReplicateQC concordance 
        # --running_mode sge 
        # --metadata_pairs examples/metadata.pairs 
        # --outdir examples/output 
        # --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep
        cmd = ["3DChromatin_ReplicateQC", "concordance", 
            "--metadata_pairs", 'metadata_pairs.txt',
            "--outdir", './',
            "--methods", "GenomeDISCO,HiCRep,HiC-Spector"] # ,QuASAR-Rep
        process = subprocess.Popen(cmd, cwd=script_work_dir)

        # 3DChromatin_ReplicateQC summary 
        # --running_mode sge 
        # --metadata_samples examples/metadata.samples 
        # --metadata_pairs examples/metadata.pairs 
        # --bins examples/Bins.w50000.bed.gz 
        # --outdir examples/output 
        # --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep    script_work_dir = './'
        '''cmd = ["3DChromatin_ReplicateQC", "summary", 
            "--running_mode", data_fp, 
            "--metadata_samples", output_path, 
            "--bins", "HiCSR", 
            "--outdir", model_fp,
            "--parameters_file", str(resolution)]
        process = subprocess.run(cmd, cwd=script_work_dir)'''

if __name__ == '__main__':
    run(chromosomes = ['22'])