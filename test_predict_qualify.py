from predict import predict

import qualify
chromosome = '19'
predict(path='./data',
        raw_path='raw',
        raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
        chromosome=chromosome,
        scale=4,
        len_size=200,
        sr_path='output',
        genomic_distance=2000000,
        start=None, end=None, draw_out=True)

input_path = qualify.configure(path='./data',
                               raw_path='raw',
                               raw_file='Rao2014-GM12878-DpnII-allreps-filtered.10kb.cool',
                               chromosome=chromosome,
                               genomic_distance=2000000,
                               resolution=10000)
file1 = input_path+'/demo_contact_true.gz'
file2 = input_path+'/demo_contact_predict.gz'
output = input_path+'/demo_scores.txt'
bedfile = input_path+'/demo.bed.gz'
script = './utils/hicrep_wrapper.R'
h_list = [20, 40, 60, 80]
for h in h_list:
    print('h: ', h)
    output = input_path+'/demo_scores_'+ str(h)+'.txt'
    qualify.score_hicrep(file1=file1, file2=file2,
                     bedfile=bedfile, output=output, script=script, h=h)