reset
debug=false

if $debug; then
    command='pudb3 dna/__main__.py'
else
    command='python3 dna'
fi

$command evaluate --model dag_transformer --model-config-path ./model_configs/dag_transformer_config.json --problem regression rank --train-path ./data/complete_classification_train.json \
	--test-size 44 --split-seed 3101978347 --output-dir ./prod_results --verbose --metafeatures-type deep
