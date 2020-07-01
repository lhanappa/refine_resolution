import prepare_data
import sys

config = prepare_data.configure(len_size=int(sys.argv[2]), genomic_distance=int(sys.argv[3]))
chromosome_list = [str(sys.argv[1])]
for chri in chromosome_list:
    prepare_data.save_samples(config, chromosome=chri)