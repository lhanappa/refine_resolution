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

def qualify_hic(predict, true):
    if type(predict) is np.ndarray:
        predict_m = predict
    elif type(predict) is str:
        predict_filepath = predict
        pred_m = np.load(predict_filepath, allow_pickle=True)
    if type(true) is np.ndarray:
        true_m = true
    elif type(true) is str:
        true_filepath = true
        true_m = np.load(true_filepath, allow_pickle=True)


def configure_file(path, filename, pair_1, pair_2, sample_1_file, sample_2_file):
    """
    path: output path
    filename: output file name
    pair_1, pair_2: 
    sample_1_file, sample_2_file: input file

    metadata_samples=${d}/metadata.samples
    echo ${metadata_samples}
    printf "HIC001\t${d}/HIC001.res50000.gz\n" > "${metadata_samples}"
    printf "HIC002\t${d}/HIC002.res50000.gz\n" >> "${metadata_samples}"
    metadata_pairs=${d}/metadata.pairs
    printf "HIC001\tHIC002\n" > "${metadata_pairs}"""
    # .pairs
    file_pairs = os.path.join(path, filename+'.pairs')
    with open(file_pairs, 'w+') as fin:
        txt = str(pair_1) + "\t" + str(pair_2)
        fin.write(txt)
    fin.close()
    # .samples
    file_samples = os.path.join(path, filename+'.samples')
    with open(file_samples, 'w+') as fin:
        txt = str(pair_1) + "\t" + str(sample_1_file) + "\n"
        txt += str(pair_2) + "\t" + str(sample_2_file) + "\n"
        fin.write(txt)
    fin.close()

    file_parameters = os.path.join(path, filename+'_parameters.txt')
    with open(file_parameters, 'w+') as fin:
        txt = """
GenomeDISCO|subsampling	lowest
GenomeDISCO|tmin	3
GenomeDISCO|tmax	3
GenomeDISCO|norm	sqrtvc
GenomeDISCO|scoresByStep	no
GenomeDISCO|removeDiag	yes
GenomeDISCO|transition	yes
HiCRep|h	20
HiCRep|maxdist	2000000
HiC-Spector|n	20
QuASAR|rebinning	resolution
SGE|text	"-l h_vmem=10G"
slurm|text	"--mem 3G
                """
        fin.write(txt)
    fin.close()