

import torch
from fe_idnn import IDNN_extractor
# from src.fe_idnn import IDNN_extractor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# set path   
image_source_path = "./dev/dev_data/images"
feature_save_path = "./dev/dev_outp"




# feature extraction and dim reduction
fe001 = IDNN_extractor(model_tag = "vgg16")
def test_001():
    assert fe001.model_tag == "vgg16"

# aaa = fe.create("layer2.3.conv3")
# aaa = fe.create("features.28")



fe = IDNN_extractor(model_tag = "ResNet50")
def test_002():
    assert fe.model_tag == "ResNet50"

fe.create("layer2.3.conv3")

fe.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 16, n_batches = 2, ecut = 1)
fe.reduce_dimension(n_neigh = 10, reduced_dim = 12)
print(fe.N.shape, fe.X.shape, fe.X_red.shape, fe.X_2D.shape,)

def test_01():
    assert fe.N.shape == (64,)

def test_02():
    assert fe.X.shape == (64, 3584)

def test_03():
    assert fe.X_red.shape == (64, 12)

def test_04():
    assert fe.X_2D.shape == (64, 2)




