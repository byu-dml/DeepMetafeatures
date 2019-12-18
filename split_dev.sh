#!/bin/bash

# small has 11 datasets
raw_data_path=./data/small_classification.tar.xz
test_size=2
test_split_seed=9232859745

python3 -m dna split-data \
    --data-path $raw_data_path \
    --test-size $test_size \
    --split-seed $test_split_seed
