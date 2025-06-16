#--------------------             
# Author : Serge Zaugg
# Description : A short script to illustrate usage of utils_ml.FeatureExtractor
#--------------------

import torch
from utils_ml import FeatureExtractor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# set path   
image_source_path = "./dev_data/images"

# feature extraction and dim reduction
fe = FeatureExtractor(model_tag = "ResNet50")
fe.eval_nodes
fe.create("layer1.2.conv3")
fe.extract(image_source_path, freq_pool = 4, batch_size = 16, n_batches = 50)
fe.save_full_features() # only needed if fe.extract was interrupted
fe.reduce_dimension(n_neigh = 10, reduced_dim = 8)

# explore resulting arrays
print(fe.N.shape, fe.X.shape, fe.X_red.shape, fe.X_2D.shape,)
fe.plot_full_features(n=80).show()
fe.plot_reduced_features(n=80).show()








