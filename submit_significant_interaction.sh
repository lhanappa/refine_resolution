#!/bin/sh
# chr=('1' '2' '3' '4' '5' '6' '7' '8' '9' '10' '11' '12' '13' '14' '15' '16' '17' '18' '19' '20' '21' '22' 'X')
chr=('17' '18' '19' '20' '21' '22' 'X')

for c in "${chr[@]}"; do
    echo sbatch bash_significant_interaction.sh ${c}
    sbatch bash_significant_interaction.sh ${c}
done