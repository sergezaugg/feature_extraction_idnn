# --------------
# Author : 
# Description : 
# --------------

import torch

# devel
from src.fe_idnn.tools import FeatureExtractor

# usage (must 'pip install' first)
from fe_idnn.tools import FeatureExtractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# set path   
image_source_path = "./dev_data/images"

# feature extraction and dim reduction
fe = FeatureExtractor(model_tag = "ResNet50")
fe.model

fe.eval_nodes
fe.create("layer1.2.conv3")
fe.create("layer2.3.conv3")
fe.create("layer3.5.conv3")
fe.create("layer4.2.conv3")
fe.extractor

fe.extract(image_source_path, freq_pool = 4, batch_size = 16, n_batches = 2, ecut = 1)

fe.extract(image_source_path, fe_save_path = "C:/Users/sezau/Downloads/aaa",  freq_pool = 4, batch_size = 16, n_batches = 2, ecut = 1)

fe.save_full_features() # only needed if fe.extract was interrupted

fe.reduce_dimension(n_neigh = 10, reduced_dim = 8)

# explore resulting arrays
print(fe.N.shape, fe.X.shape, fe.X_red.shape, fe.X_2D.shape,)
fe.plot_full_features(n=20).show()
fe.plot_reduced_features(n=20).show()







