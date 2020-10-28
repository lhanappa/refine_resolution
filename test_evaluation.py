import sys
import os
from our_model import qualify
from our_model.utils import operations

def evaluate_hicrep(chromosomes, methods, input_path='./experiment/evaluation/'):
    root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
    for chro in chromosomes:
        for method in methods:
            file1 = os.path.join(input_path, 'chr{}'.format(chro), 'high_chr{}_contact.gz'.format(chro))
            file2 = os.path.join(input_path, 'chr{}'.format(chro), '{}_chr{}_contact.gz'.format(method, chro))
            m1name = 'true_{}'.format(chromosome)
            m2name = '{}_{}'.format(method, chromosome)
            bedfile = os.path.join(input_path, 'chr{}'.format(chro), 'bins_chr{}.bed.gz'.format(chro))
            script = os.path.join(root_dir, 'our_model', 'utils','hicrep_wrapper.R')
            h_list = [20]#, 40, 60, 80]
            for h in h_list:
                print('h: ', h)
                output = os.path.join(input_path, 'chr{}'.format(chro), '{}_chr{}_hicrep_{}.txt'.format(method, chro, h))
                qualify.score_hicrep(file1=file1, file2=file2,
                                bedfile=bedfile, output_path=output, script=script, h=h,
                                m1name=m1name, m2name=m2name)

if __name__ == '__main__':
    evaluate_hicrep(['22'], ['ours'])

"""root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
len_size = 200
max_dis = 2000000
lr_size = 40
hr_size = 40
model = str(sys.argv[1]) # deephic, hicgan, hicsr
chromosome = str(sys.argv[2])
if model == 'hicsr':
    hr_size = 28

input_path,sr_file = qualify.configure_model(model=model, path=os.path.join(root_dir, 'data'),
                               raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
                               sr_path = '_'.join(['output', model, str(max_dis), str(lr_size), str(hr_size)]),
                               chromosome=chromosome,
                               genomic_distance=2000000,
                               resolution=10000, 
                               true_path='_'.join(['output', 'ours',str(max_dis), str(len_size)]))

file1 = os.path.join(input_path, sr_file+'_contact_true.gz')
file2 = os.path.join(input_path, sr_file+'_contact_predict.gz')
m1name = 'true_{}'.format(chromosome)
m2name = 'predict_{}'.format(chromosome)
output_path = os.path.join(input_path, sr_file+'_scores')
bedfile = os.path.join(input_path, sr_file+'.bed.gz')


script = os.path.join(root_dir, 'our_model', 'utils','hicrep_wrapper.R')
h_list = [20]#, 40, 60, 80]
for h in h_list:
    print('h: ', h)
    output = output_path+ str(h)+'.txt'
    qualify.score_hicrep(file1=file1, file2=file2,
                     bedfile=bedfile, output_path=output, script=script, h=h,
                     m1name=m1name, m2name=m2name)"""