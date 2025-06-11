#--------------------             
# Author : Serge Zaugg
# Description : A script to keep track of and reproduce feature extraction history
# Mainly to apply pt_extract_features.utils_ml.FeatureExtractor with several models and params
#--------------------

import torch
from pt_extract_features.utils_ml import FeatureExtractor
torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#-------------------------
# set paths   
image_path = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
featu_path = "./extracted_features"
n_batches = 10 # dev
# n_batches = 800000 # prod


fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer1.2.conv3")
fe.extract(image_path, featu_path, freq_pool = 4, batch_size = 16, n_batches = n_batches)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]


fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer2.3.conv3")
fe.extract(image_path, featu_path, freq_pool = 4, batch_size = 16, n_batches = n_batches)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]


fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer3.5.conv3")
fe.extract(image_path, featu_path, freq_pool = 4, batch_size = 16, n_batches = n_batches)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]


fe = FeatureExtractor(model_tag = "DenseNet121")
fe.eval_nodes
fe.create("features.denseblock3")
fe.extract(image_path, featu_path, freq_pool = 4, batch_size = 16, n_batches = n_batches)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]


fe = FeatureExtractor(model_tag = "vgg16")
fe.eval_nodes
fe.create("features.28")
fe.extract(image_path, featu_path, freq_pool = 4, batch_size = 16, n_batches = n_batches)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]


fe = FeatureExtractor(model_tag = "MaxVit_T")
fe.eval_nodes
fe.create("blocks.3.layers.1.layers.MBconv.layers.conv_c")
fe.extract(image_path, featu_path, freq_pool = 1, batch_size = 16, n_batches = n_batches)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]




