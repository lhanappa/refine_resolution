
chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' 'X')
#chr=('22')
#contact=('1' '2' '3' '4' '5' '6' '7' '8')

rm slurm-data-*.out
for c in "${chr[@]}"; do
    sbatch bash_prepare_data.sh $c
done
