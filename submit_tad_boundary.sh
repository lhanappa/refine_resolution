#!/bin/sh
chr=('19' '20' '21' '22' 'X')
# method=('ours_80' 'ours_200' 'ours_400' 'deephic_40' 'hicsr_40')
method=('ours_400' 'deephic_40' 'hicsr_40' 'low')

#rm slurm-data-*.out
for m in "${method[@]}"; do
    echo sbatch bash_gather_tad_boundary.sh ${m}
    sbatch bash_gather_tad_boundary.sh ${m}
done

# for c in "${chr[@]}"; do
#    for m in "${method[@]}"; do
#        echo sbatch bash_tad_boundary.sh ${m}
#        sbatch bash_tad_boundary.sh ${m}
#    done
# done