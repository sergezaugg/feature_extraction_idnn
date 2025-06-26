#--------------------             
# Author : Serge Zaugg
# Description : A simple flat-script to batch-extract features from one dataset with several models 
#--------------------

import torch
from fe_idnn import IDNN_extractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Set paths  
image_source_path = "./dev/dev_data/images"
feature_save_path = "./dev/dev_outp"
n_batches = 10 # set n_batches to very large value to process all image in a real run

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer1.2.conv3")
fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer2.3.conv3")
fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer3.5.conv3")
fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "ResNet50")
fe.create("layer4.2.conv3")
fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 0)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "DenseNet121")
fe.create("features.denseblock3")
fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "vgg16")
fe.create("features.28")
fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

fe = IDNN_extractor(model_tag = "MaxVit_T")
fe.create("blocks.3.layers.1.layers.MBconv.layers.conv_c")
fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 1, batch_size = 16, n_batches = n_batches, ecut = 1)
[fe.reduce_dimension(n_neigh = 10, reduced_dim = d) for d in [2,4,8,16]]

