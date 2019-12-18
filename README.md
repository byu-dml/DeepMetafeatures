# Deep Metafeatures

This repository contains a metafeature model that takes in a tensor version of a data set which it extracts a vector representation of using deep learning. This vector serves ideally act as metafeatures which are more informative than traditional metafeatures alone. The model as a whole contains the metafeature model and a meta model which are trained and evaluated together as one.

## Instructions for use:
0. Setup the python enviroment (`pip3 install -r requirements.txt`)
1. bash split_dev.sh to untar the development data set and split it into training and test sets
2. bash split_prod.sh to do the same with the production data set
3. bash development.sh to train and evaluate the model on the development data set
4. bash production.sh to do the same with the production data set
5. bash tuning.sh to tune the model's hyper parameters

## Configuration and results
1. In production.sh, development.sh, and tuning.sh, you can change which model is being used. The models to choose from are lstm, dag_lstm, transformer, and dag_transformer
2. In tuning.sh, you can select whether to use the development data set or the production data set since tuning is not separated into two bash scripts, on for development and the other for production
3. Additionally, in tuning.sh, you can select which tuning algorithm to use, whether it's bayesian or a genetic algorithm.
4. In development.sh, tuning.sh, and production.sh, you can also choose which metafeature type to use. The choices are traditional, deep, or both.
