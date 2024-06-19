#!/bin/bash
echo "Completed..."
for dataset in test train combined
do
    for classifier in RF MLP SVM regression
    do
        for labels in remove_0 remove_6 remove_0_6 misc_class
        do
            python classifier_baselines.py \
                --dataset $dataset \
                --boundary_classes $labels \
                --classifier $classifier \
                --use_basics True \
                --use_benepar True \
                --use_amr True
            echo "$dataset $classifier $labels." 
        done
    done
done

for dataset in test train combined
do
    for classifier in RF MLP SVM regression
    do
        python classifier_baselines.py \
            --dataset $dataset \
            --classifier $classifier \
            --use_basics True \
            --use_benepar True \
            --use_amr True
        echo "$dataset $classifier without boundary label filter."
    done
done