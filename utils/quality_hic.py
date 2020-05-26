"""
call 3DChromatin_ReplicateQC to qualify Hi-C matrix
"""

"""
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
    

def format_contact(matrix, coordinate=(0,1),resolution=10000, chrm='1', save_file=True, filename=None):
    """chr1 bin1 chr2 bin2 value"""
    n = len(matrix)
    nhf = int(len(matrix)/2)
    contact = list()
    for i in np.arange(n):
        for j in np.arange(i+1, n):
            value = float(matrix[i,j])
            if value <= 1.0e-10:
                continue
            chr1 = chrm
            chr2 = chrm
            bin1 = (i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution
            bin2 = (j - int(j/nhf)*nhf + coordinate[int(j/nhf)]*nhf)*resolution
            entry = [chr1, bin1, chr2, bin2, value]
            contact.append(entry)
    if save_file:
        if filename is None:
            filename = './demo_contact.gz'
        np.savetxt(filename, contact)
    return contact

def format_bin(matrix, coordinate=(0,1),resolution=10000, chrm='1', save_file=True, filename=None):
    """chr start end name"""
    n = len(matrix)
    nhf = int(len(matrix)/2)
    bins = list()
    for i in np.arange(n):
        chr1 = chrm
        start = (i - int(i/nhf)*nhf + coordinate[int(i/nhf)]*nhf)*resolution
        end = start + resolution
        entry = [chr1, start, end, start]
        bins.append(entry)
    if save_file:
        if filename is None:
            filename = './demo.bed'
        np.savetxt(filename, bins)
    return bins

def configure_file(path, filename, pair_1, pair_2, sample_1_file, sample_2_file):
    """
    metadata_samples=${d}/metadata.samples
    echo ${metadata_samples}
    printf "HIC001\t${d}/HIC001.res50000.gz\n" > "${metadata_samples}"
    printf "HIC002\t${d}/HIC002.res50000.gz\n" >> "${metadata_samples}"
    metadata_pairs=${d}/metadata.pairs
    printf "HIC001\tHIC002\n" > "${metadata_pairs}"""
    #.pairs
    file_pairs = filename+'.pairs'
    txt = str(pair_1) + "\t" + str(pair_2)
    np.savetxt(path+file_pairs, txt)
    #.samples
    file_samples = filename+'.samples'
    txt = str(pair_1) + "\t" + str(sample_1_file) + "\n"
    txt += str(pair_2) + "\t" + str(sample_2_file) + "\n"
    np.savetxt(path+file_samples, txt)


"""
GenomeDISCO|subsampling	lowest
GenomeDISCO|tmin	3
GenomeDISCO|tmax	3
GenomeDISCO|norm	sqrtvc
GenomeDISCO|scoresByStep	no
GenomeDISCO|removeDiag	yes
GenomeDISCO|transition	yes
HiCRep|h	5
HiCRep|maxdist	5000000
HiC-Spector|n	20
QuASAR|rebinning	resolution
SGE|text	"-l h_vmem=10G"
slurm|text	"--mem 3G
""""