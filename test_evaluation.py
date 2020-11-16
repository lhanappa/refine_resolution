import sys
import os
from our_model import qualify
from our_model.utils import operations
import numpy as np

def evaluate_hicrep(chromosomes, methods, input_path='./experiment/evaluation/'):
    root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
    for chro in chromosomes:
        for method in methods:
            file1 = os.path.join(input_path, 'chr{}'.format(chro), 'high_chr{}_contact.gz'.format(chro))
            file2 = os.path.join(input_path, 'chr{}'.format(chro), '{}_chr{}_contact.gz'.format(method, chro))
            m1name = 'true_{}'.format(chro)
            m2name = '{}_{}'.format(method, chro)
            bedfile = os.path.join(input_path, 'chr{}'.format(chro), 'bins_chr{}.bed.gz'.format(chro))
            script = os.path.join(root_dir, 'our_model', 'utils','hicrep_wrapper.R')
            h_list = np.append(np.arange(5,12), 20)
            print(h_list)
            output_path = os.path.join(input_path, 'chr{}'.format(chro), 'metrics')
            os.makedirs(output_path, exist_ok=True)
            summary_file = os.path.join(output_path,'{}_chr{}_hicrep.txt'.format(method, chro))
            for h in h_list:
                print('h: ', h)
                lines = list()
                output = os.path.join(output_path,'{}_chr{}_hicrep_{}.txt'.format(method, chro, h))
                qualify.score_hicrep(file1=file1, file2=file2,
                                bedfile=bedfile, output_path=output, script=script, h=h,
                                m1name=m1name, m2name=m2name)
                with open(output, 'r') as fin:
                    for line in fin:
                        lines.append(line)
                    fin.close()

                with open(summary_file, 'a') as fout:
                    for line in lines:
                        l = line.split()
                        m0 = l[0].split('_')[0]
                        m1 = l[1].split('_')[0]
                        chro = l[0].split('_')[1]
                        score = l[2]
                        sd = l[3]
                        line = 'chr{}\t{}\t{}\t{}\t{}\t{}\n'.format(chro, h, m0, m1, score, sd)
                        fout.write(line)
                    fout.close()
            for h in h_list:
                output = os.path.join(output_path,'{}_chr{}_hicrep_{}.txt'.format(method, chro, h))
                if os.path.exists(output):
                    os.remove(output)
                

def evaluate_mae(chromosomes, methods, input_path='./experiment/evaluation/', max_boundary=200, diag_k=2):
    # root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
    for chro in chromosomes:
        for method in methods:
            path = os.path.join(input_path, 'chr{}'.format(chro))
            files = [f for f in os.listdir(path) if (chro in f and '.npz' in f)]
            file1 = [f for f in files if 'high' in f][0]
            file1 = os.path.join(path, file1)
            file2 = [f for f in files if method in f][0]
            file2 = os.path.join(path, file2)
            m1name = 'high_{}'.format(chro)
            m2name = '{}_{}'.format(method, chro)
            print(file1)
            print(file2)
            output_path = os.path.join(input_path, 'chr{}'.format(chro), 'metrics')
            os.makedirs(output_path, exist_ok=True)
            output = os.path.join(output_path, '{}_chr{}_mae.txt'.format(method, chro))
            qualify.metric_mae(file1=file1, file2=file2, model=method, output_path=output,
                            m1name=m1name, m2name=m2name, max_boundary=max_boundary, diag_k=diag_k)

def evaluate_mse(chromosomes, methods, input_path='./experiment/evaluation/', max_boundary=200, diag_k=2):
    # root_dir = operations.redircwd_back_projroot(project_name='refine_resolution')
    for chro in chromosomes:
        for method in methods:
            path = os.path.join(input_path, 'chr{}'.format(chro))
            files = [f for f in os.listdir(path) if (chro in f and '.npz' in f)]
            file1 = [f for f in files if 'high' in f][0]
            file1 = os.path.join(path, file1)
            file2 = [f for f in files if method in f][0]
            file2 = os.path.join(path, file2)
            m1name = 'high_{}'.format(chro)
            m2name = '{}_{}'.format(method, chro)
            print(file1)
            print(file2)
            output_path = os.path.join(input_path, 'chr{}'.format(chro), 'metrics')
            os.makedirs(output_path, exist_ok=True)
            output = os.path.join(output_path, '{}_chr{}_mse.txt'.format(method, chro))
            qualify.metric_mse(file1=file1, file2=file2, model = method, output_path=output,
                            m1name=m1name, m2name=m2name, max_boundary=max_boundary, diag_k=diag_k)

if __name__ == '__main__':
    #cool_file = 'Rao2014-GM12878-MboI-allreps-filtered.10kb.cool'
    cool_file = 'Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool'
    cell_type = cool_file.split('_')[0] + '_' + cool_file.split('_')[1] + '_' + cool_file.split('_')[2] + '_' + cool_file.split('.')[1]
    destination_path = os.path.join('./experiment/evaluation/', cell_type)

    # models = [str(sys.argv[1])] # deephic, hicgan, hicsr, ours
    chromosomes = [str(sys.argv[1])] # 22, 21, 20, 19, X

    # evaluate_hicrep([chromosome], [model])
    # chromosomes = ['22', '21', '20', '19', 'X']
    # chromosomes = ['22']
    models = ['deephic', 'hicgan', 'hicsr', 'ours']
    evaluate_hicrep(chromosomes, models, input_path=destination_path)
    
    # chromosomes = ['22', '21', '20', '19', 'X']
    # chromosomes = ['22']
    # models = ['deephic', 'hicgan', 'hicsr', 'ours', 'low']
    # evaluate_mae(chromosomes, models, input_path=destination_path)
    # evaluate_mse(chromosomes, models, input_path=destination_path)
    
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