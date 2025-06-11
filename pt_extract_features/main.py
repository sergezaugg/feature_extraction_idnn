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
fe.extract(image_path, featu_path, freq_pool = 4, batch_size = 16, n_batches = 5)
fe.save_full_features() # only needed if fe.extract was interrupted
fe.reduce_dimension(n_neigh = 10, reduced_dim = 16)


fe.plot_full_features(n=80).show()
fe.plot_reduced_features(n=80).show()


fe.out_name
fe.out_name_reduced
print(fe.N.shape, fe.X.shape, fe.X_red.shape, fe.X_2D.shape,)


 




# apply to older objects
old_path = "extracted_features/20250610_173049_full_features_ResNet50_layer1.2.conv3.npz"
fe.reduce_dimension(old_path, n_neigh = 10, reduced_dim = [2,4,8,16])
old_path = "extracted_features/20250610_173105_dimred_16_neigh_10_ResNet50_layer1.2.conv3.npz"
fe.plot_reduced_features(npzfile_reduced_path = old_path).show()

