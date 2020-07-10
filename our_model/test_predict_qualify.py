from predict import predict
import sys
import os
import qualify
from utils import operations
chromosome = str(sys.argv[1])

root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
len_size = 200
max_dis = 2000000
predict(path=os.path.join(root_dir, 'data'),
        raw_path='raw',
        raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        chromosome=chromosome,
        scale=4,
        len_size=200,
        sr_path='_'.join(['output','ours',str(max_dis), str(len_size)]),
        genomic_distance=2000000,
        start=None, end=None, draw_out=True)

input_path,_ = qualify.configure_our_model(path=os.path.join(root_dir, 'data'),
                               raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
                               sr_path = '_'.join(['output','ours',str(max_dis), str(len_size)]),
                               chromosome=chromosome,
                               genomic_distance=2000000,
                               resolution=10000)
file1 = input_path+'/demo_contact_true.gz'
file2 = input_path+'/demo_contact_predict.gz'
output = input_path+'/demo_scores.txt'
bedfile = input_path+'/demo.bed.gz'
script = './utils/hicrep_wrapper.R'
h_list = [20]#, 40, 60, 80]
for h in h_list:
    print('h: ', h)
    output = input_path+'/demo_scores_'+ str(h)+'.txt'
    qualify.score_hicrep(file1=file1, file2=file2,
                     bedfile=bedfile, output=output, script=script, h=h)