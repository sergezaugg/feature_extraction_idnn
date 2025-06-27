# --------------
# Author : 
# Description : 
# --------------

import torch
from src.fe_idnn import IDNN_extractor
# # usage (must 'pip install' first)
# from fe_idnn import IDNN_extractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# set path   
image_source_path = "./dev/dev_data/images"
feature_save_path = "./dev/dev_outp"

# feature extraction and dim reduction
fe = IDNN_extractor(model_tag = "ResNet50")
fe.model

# "vgg16"
# fe.model_tag

fe.eval_nodes
fe.create("layer2.3.conv3")
fe.extractor

fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = 2, ecut = 1)
fe.save_full_features() # only needed if fe.extract was interrupted
fe.reduce_dimension(n_neigh = 10, reduced_dim = 12)
# explore resulting arrays
print(fe.N.shape, fe.X.shape, fe.X_red.shape, fe.X_2D.shape,)








