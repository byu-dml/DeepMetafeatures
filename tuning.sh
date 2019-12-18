#!/bin/bash

reset
use_complete_data=false
debug=false
# rmse, pearson, spearman, ndcg, ndcg_at_k, regret, regret_at_k
objective=spearman
model_seed=10

if $use_complete_data; then
    # complete has 194 datasets
    train_path=./data/complete_classification_train.json
    validation_size=25
    validation_split_seed=3101978347
    metafeature_subset=all
    results_dir=./results
    tuning_output_dir=./tuning_output

else
    # small has 11 datasets
    train_path=./data/small_classification_train.json
    validation_size=2
    validation_split_seed=5460650386
    metafeature_subset=all
    results_dir=./dev_results
    tuning_output_dir=./dev_tuning_output
fi

model=dag_transformer
n_generations=2
population_size=2
# tuning_type=genetic
tuning_type=bayesian
n_calls=5
# warm_start="path/to/warm/start.csv"

if $debug; then
    command='pudb3 dna/__main__.py'
else
    command='python3 -m dna'
fi

$command tune \
    --model $model \
    --model-config-path ./model_configs/${model}_config.json \
    --tuning-config-path ./tuning_configs/${model}_tuning_config.json \
    --tuning-output-dir $tuning_output_dir \
    --problem regression rank \
    --objective $objective \
    --train-path $train_path \
    --metafeature-subset $metafeature_subset \
    --test-size $validation_size \
    --split-seed $validation_split_seed \
    --model-seed $model_seed \
    --output-dir $results_dir \
    --n-generations $n_generations \
    --population-size $population_size \
    --tuning-type $tuning_type \
    --n-calls $n_calls \
    --verbose \
    --metafeatures-type deep
    # --warm-start-path $warm_start \
