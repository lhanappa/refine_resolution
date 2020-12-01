chr=('19' '20' '21' '22' 'X')
method=('ours_80' 'ours_200' 'ours_400' 'deephic_40' 'hicsr_40')

#rm slurm-data-*.out
for c in "${chr[@]}"; do
    for m in "${method[@]}"; do
        echo sbatch bash_evaluation.sh ${c} ${m}
        sbatch bash_evaluation.sh ${c} ${m}
    done
done