#!/bin/bash

for classifier in "RF" "MLP" "SVM"
do
    for value in "True" "False"
    do
        python3 classifier_baselines.py \
            --classifier $classifier \
            --dataset "train" \
            --clean $value \
            --add_embeddings False
    done
done