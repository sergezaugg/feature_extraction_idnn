#--------------------             
# Author : Serge Zaugg
# Description : A short script to illustrate usage of pt_extract_features.utils_ml.FeatureExtractor and dim_reduce
#--------------------

import torch
from pt_extract_features.utils_ml import FeatureExtractor, dim_reduce
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# set paths   
image_path = "D:/xc_real_projects/xc_sw_europe/xc_spectrograms"
featu_path = "./extracted_features"

# feature extraction
fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer1.2.conv3")
fe.extract(image_path, freq_pool = 4, batch_size = 16, n_batches = 5)
fe.save_full_features(featu_path)
print(fe.X.shape, fe.N.shape,)
fe.reduce_dimension(featu_path, n_neigh = 10, reduced_dim = [2,4,8,16])


