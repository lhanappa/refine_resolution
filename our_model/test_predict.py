from predict import predict
import sys
import os
from utils import operations

raw_list = ['Rao2014-CH12LX-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool', 
        'Rao2014-HMEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-HUVEC-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-IMR90-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-K562-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-KBM7-MboI-allreps-filtered.10kb.cool', 
        'Rao2014-NHEK-MboI-allreps-filtered.10kb.cool']

# 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool',

root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
# raw_hic='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
# raw_hic = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
chromosome = ['1', '2', '3', '4', '5', '6', '7', '8', \
                '9', '10', '11', '12', '13', '14', '15', '16', \
                '17', '18', '19', '20', '21', '22', 'X'] # str(sys.argv[1])
file_idx = int(sys.argv[1])
len_size = int(sys.argv[2]) # 200
raw_hic = raw_list[file_idx]
max_dis = 2000000

model_path = os.path.join(root_dir, 'trained_model', 'enhic', 'saved_model', 'gen_model_' + str(len_size), 'gen_weights')
chromosome.reverse()
for chro in chromosome:
        predict(path=os.path.join(root_dir, 'data'),
                raw_path='raw',
                raw_file=raw_hic,
                model_path=None,
                chromosome=chro,
                scale=4,
                len_size=len_size,
                sr_path='_'.join(['output','ours',str(max_dis), str(len_size)]),
                genomic_distance=2000000,
                start=None, end=None, draw_out=True)