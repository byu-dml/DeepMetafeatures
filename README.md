# Deep Metafeatures

This repository contains a metafeature model that takes in a tensor version of a data set which it extracts a vector representation of using deep learning. This vector serves to ideally act as metafeatures which are more informative than traditional metafeatures alone. The model as a whole contains the metafeature model and a meta model which are trained and evaluated together as one.

## Instructions for use:
0. Setup the python enviroment (`pip3 install -r requirements.txt`)
1. bash split_dev.sh to untar the development data set and split it into training and test sets
2. bash split_prod.sh to do the same with the production data set
3. bash development.sh to train and evaluate the model on the development data set
4. bash production.sh to do the same with the production data set
5. bash tuning.sh to tune the model's hyper parameters

## Configuration and results
1. In production.sh, development.sh, and tuning.sh, you can change which meta model is being used. The meta models to choose from are lstm, dag_lstm, transformer, and dag_transformer
2. In tuning.sh, you need to select whether to use the development data set or the production data set since tuning is **not** separated into two bash scripts, one for development and the other for production
3. Additionally, in tuning.sh, you can select which tuning algorithm to use, whether it's bayesian or a genetic algorithm.
4. In development.sh, production.sh, and tuning.sh, you can also choose which metafeature type to use. The choices are traditional, deep, or both.

## The contributions of this repo are as follows:
1. In dna/data.py lines 226-312, there is code for preprocessing the pipelines such that irrelevant primitives are removed. These primitives can only hurt the meta model's ability to learn. While there is a possibility that they may cause a neutral reaction, they certainly can't help, so we're better off without them. After they are removed, the pipelines with their remaining primitives are reconstructed in a way that makes sense. Remember you need to remove .cache and re-create it!
2. Two new meta models. One is a transformer and the other is a DAG transformer. Both of these are entirely parallelized (no looping) and they use attention to combine the pipelines with the meta features rather than just concatenating them and sending them through a linear layer. The DAG transformer's batches can be grouped by length instead of structure which is highly useful. However, PyTorch's transformer's can only take in one mask. But PyTorch is planning to update such that you can have customized masks. Before they do that though, the temporary fix is to go into your virtual environment directory and then into lib/python3.6/site-packages/torch/nn, replacing python3.6 with whatever version you have. Then in functional.py, right under line 3350 where it says "if attn_mask is not None:", insert the following two lines:
* if len(attn_mask.shape) < 3:
* (indent) attn_mask = attn_mask.unsqueeze(0)
3. The deep meta feature model trains along side a meta model to learn how to produce the best meta features specifically for that model. There certainly could be improvements made to it. But keep in mind that if the deep meta feature model is used, only one data set can be passed through the meta model at a time, which means that the batches need to be grouped by data set. It might be easier to try auto-encoding data sets instead, but that would have both benefits and costs.
4. A new loss function called Pearson loss which uses the inverse of the Pearson correlation on a batch of outputs. This is a list wise loss and requires, again, that all the outputs refer to the same data set. Otherwise the correlation doesn't make any sense.
