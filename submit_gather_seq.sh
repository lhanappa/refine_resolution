#!/bin/bash -l
SCALE=${1} # 4 8 16 32 48 64
chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' 'X' '15' '16' '17' '18' '19' '20' '21' '22') #  
for c in "${chr[@]}"; do
    echo sbatch bash_gather_seq.sh ${SCALE} ${c}
    sbatch bash_gather_seq.sh ${SCALE} ${c}
done

# source activate env_fr_test
# SCALE=${1} # 4,8,16,32,48,64
# echo python evaluation_gather_data_seq.py ${SCALE}
# python evaluation_gather_data_seq.py ${SCALE}