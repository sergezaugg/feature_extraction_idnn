#--------------------             
# Author : Serge Zaugg
# Description : A short script to illustrate usage 
#--------------------

import torch
from fe_idnn.tools import IDNN_extractor
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# set path   
image_source_path = "./dev_data/images"
# initialize instance of feature extraction and dim reduction
fe = IDNN_extractor(model_tag = "ResNet50")
# (optional) check model architecture
fe.model
fe.eval_nodes
# create the feature extractor
fe.create("layer3.5.conv3")
# extract features from images
fe.extract(image_source_path, freq_pool = 4, batch_size = 16, n_batches = 2, ecut = 1)
fe.save_full_features() # only needed if fe.extract was interrupted
# apply UMAP to reduce dim
fe.reduce_dimension(n_neigh = 10, reduced_dim = 8)
# (optional) explore resulting arrays
print(fe.N.shape, fe.X.shape, fe.X_red.shape, fe.X_2D.shape,)
fe.plot_full_features(n=20).show()
fe.plot_reduced_features(n=20).show()








