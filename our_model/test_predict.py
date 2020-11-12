from predict import predict
import sys
import os
from utils import operations
chromosome = str(sys.argv[1])

root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
# raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
len_size = 200
max_dis = 2000000
predict(path=os.path.join(root_dir, 'data'),
        raw_path='raw',
        raw_file=raw_hic,
        chromosome=chromosome,
        scale=4,
        len_size=200,
        sr_path='_'.join(['output','ours',str(max_dis), str(len_size)]),
        genomic_distance=2000000,
        start=None, end=None, draw_out=True)