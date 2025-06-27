#--------------------             
# Author : Serge Zaugg
# Description : A few basic tests for CI
#--------------------

import torch
from fe_idnn import IDNN_extractor # for pytest
# from src.fe_idnn import IDNN_extractor # for interactive dev of tests

# temp to suppress warning triggered by UMAP when using sklearn
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

# set path   
image_source_path = "./dev/dev_data/images"
feature_save_path = "./dev/dev_outp"


# test battery 1
fe001 = IDNN_extractor(model_tag = "vgg16")
fe001.create("features.28")
fe001.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 2, batch_size = 16, n_batches = 2, ecut = 2)
fe001.reduce_dimension(n_neigh = 15, reduced_dim = 16)
print(fe001.N.shape, fe001.X.shape, fe001.X_red.shape, fe001.X_2D.shape,)
def test_0011():
    assert fe001.model_tag == "vgg16"

def test_0012():
    assert hasattr(fe001, 'extractor')

def test_0013():
    assert fe001.N.shape == (32,)

def test_0014():
    assert fe001.X.shape == (32, 3584)

def test_0015():
    assert fe001.X_red.shape == (32, 16)

def test_0016():
    assert fe001.X_2D.shape == (32, 2)


# test battery 2
fe002 = IDNN_extractor(model_tag = "ResNet50")
fe002.create("layer3.5.conv3")
fe002.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 4, batch_size = 8, n_batches = 3, ecut = 1)
fe002.reduce_dimension(n_neigh = 10, reduced_dim = 12)
print(fe002.N.shape, fe002.X.shape, fe002.X_red.shape, fe002.X_2D.shape,)

def test_0021():
    assert fe002.model_tag == "ResNet50"

def test_0022():
    assert hasattr(fe002, 'extractor')

def test_0023():
    assert fe002.N.shape == (24,)

def test_0024():
    assert fe002.X.shape == (24, 4096)

def test_0025():
    assert fe002.X_red.shape == (24, 12)

def test_0026():
    assert fe002.X_2D.shape == (24, 2)


# test battery 3
fe003 = IDNN_extractor(model_tag = "DenseNet121")
fe003.create("features.denseblock3")
fe003.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 1, batch_size = 2, n_batches = 4)
print(fe003.N.shape, fe003.X.shape)
def test_0031():
    assert fe003.N.shape == (8,)
    assert fe003.X.shape == (8, 14336)

fe004 = IDNN_extractor(model_tag = "DenseNet121")
fe004.create("features.denseblock3")
fe004.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 2, batch_size = 4, n_batches = 3, ecut = 1)
print(fe004.N.shape, fe004.X.shape)
def test_0032():
    assert fe004.N.shape == (12,)
    assert fe004.X.shape == (12, 7168)

fe005 = IDNN_extractor(model_tag = "DenseNet121")
fe005.create("features.denseblock3")
fe005.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 3, batch_size = 6, n_batches = 2, ecut = 2)
print(fe005.N.shape, fe005.X.shape)
def test_0033():
    assert fe005.N.shape == (12,)
    assert fe005.X.shape == (12, 5120)

fe006 = IDNN_extractor(model_tag = "DenseNet121")
fe006.create("features.denseblock3")
fe006.extract(image_path = image_source_path, fe_save_path = feature_save_path,  freq_pool = 5, batch_size = 11, n_batches = 1, ecut = 3)
print(fe006.N.shape, fe006.X.shape)
def test_0034():
    assert fe006.N.shape == (11,)
    assert fe006.X.shape == (11, 3072)
