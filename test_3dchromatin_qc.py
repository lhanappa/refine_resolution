# this file qualify hic 1 and hic 2 by calling 3Dchromatin_replicateQC
# https://github.com/kundajelab/3DChromatin_ReplicateQC
# in Python 2.7
import numpy as np
import gzip
import os, sys
import subprocess

"""GenomeDISCO|subsampling	lowest
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
"""

parameters = '''GenomeDISCO|subsampling	lowest
GenomeDISCO|tmin	1
GenomeDISCO|tmax	1
GenomeDISCO|norm	sqrtvc
GenomeDISCO|scoresByStep	yes
GenomeDISCO|removeDiag	yes
GenomeDISCO|transition	yes
HiCRep|h	3
HiCRep|maxdist	2000000
HiC-Spector|n	1
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
    print(methods)
    for file in files:
        f = os.path.join(path, file)
        absolute_path = os.path.abspath(f)
        method = '_'.join(file.split('_')[0:-2])
        if method in methods:
            print(method, absolute_path)
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
    chromosomes = ['22', '21', '20', '19', 'X'],
    cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'):

    cell_type = cool_file.split('-')[0] + '_' + cool_file.split('-')[1] + '_' + cool_file.split('-')[2] + '_' + cool_file.split('.')[1]
    destination_path = os.path.join('./experiment/evaluation/', cell_type)

    print(chromosomes)
    process = []
    for chro in chromosomes:
        generate_parameters(chro, path=destination_path)
        generate_metadata_samples(methods, chro, path=destination_path)
        generate_pairs(list1, list2, chro, path=destination_path)

        # 3DChromatin_ReplicateQC preprocess 
        # --running_mode sge 
        # --metadata_samples examples/metadata.samples 
        # --bins examples/Bins.w50000.bed.gz 
        # --outdir examples/output 
        # --parameters_file examples/example_parameters.txt
        script_work_dir = os.path.join(destination_path, 'chr{}'.format(chro))
        cmd = ["3DChromatin_ReplicateQC", "preprocess", 
            "--metadata_samples",  'metadata_samples.txt', 
            "--bins", 'bins_chr{}.bed.gz'.format(chro), 
            "--outdir", './chromatin_qc/',
            "--methods", "GenomeDISCO,HiC-Spector,HiCRep",  # HiCRep, ,QuASAR-Rep
            "--parameters_file", './qc_parameters.txt']
        process.append(subprocess.Popen(cmd, cwd=script_work_dir))
    for p in process:
        p.wait()

    for chro in chromosomes:
        # 3DChromatin_ReplicateQC concordance 
        # --running_mode sge 
        # --metadata_pairs examples/metadata.pairs 
        # --outdir examples/output 
        # --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep
        script_work_dir = os.path.join(destination_path, 'chr{}'.format(chro))
        cmd = ["3DChromatin_ReplicateQC", "concordance", 
            "--metadata_pairs", 'metadata_pairs.txt',
            "--outdir", './chromatin_qc/',
            "--methods", "GenomeDISCO,HiC-Spector,HiCRep"] # ,QuASAR-Rep
        process.append(subprocess.Popen(cmd, cwd=script_work_dir))
    for p in process:
            p.wait()

    for chro in chromosomes:
        # 3DChromatin_ReplicateQC summary 
        # --running_mode sge 
        # --metadata_samples examples/metadata.samples 
        # --metadata_pairs examples/metadata.pairs 
        # --bins examples/Bins.w50000.bed.gz 
        # --outdir examples/output 
        # --methods GenomeDISCO,HiCRep,HiC-Spector,QuASAR-Rep    script_work_dir = './'
        cmd = ["3DChromatin_ReplicateQC", "summary", 
            "--metadata_pairs", 'metadata_pairs.txt',
            "--metadata_samples", 'metadata_samples.txt', 
            "--bins", 'bins_chr{}.bed.gz'.format(chro), 
            "--methods", "GenomeDISCO,HiC-Spector,HiCRep",
            "--outdir", './chromatin_qc/']
        process.append(subprocess.Popen(cmd, cwd=script_work_dir))
    for p in process:
        p.wait()

if __name__ == '__main__':
    raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
        'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool',
        'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool']

    # 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool',6
    idx = int(sys.argv[1])
    chro = str(sys.argv[2])
    # methods = ['deephic_40', 'hicsr_40', 'ours_80', 'ours_200', 'ours_400', 'high', 'low']
    methods = ['deephic_40', 'hicsr_40', 'ours_400', 'high', 'low']
    list1 = ['high']
    list2 = ['deephic_40', 'hicsr_40', 'ours_400', 'low']
    cool_file = raw_list[idx]

    run(methods = methods, list1=list1, list2=list2, chromosomes = [chro], cool_file=cool_file)