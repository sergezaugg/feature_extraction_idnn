#--------------------             
# Author : Serge Zaugg
# Description : Functions an classes specific to ML/PyTorch backend
#--------------------

import os
import pandas as pd
import numpy as np
import torch
import datetime
from torch.utils.data import Dataset
from torchvision.io import decode_image
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from sklearn.model_selection import train_test_split
import umap.umap_ as umap
import skimage.measure
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names

class ImageDataset(Dataset):
    """
    Description: A simple PyTorch dataset (loader) to batch process images from file
    """
    def __init__(self, imgpath, preprocess):
        """
        imgpath (str) : path to a dir that contains JPG images. 
        label_path (str) : path to a csv file which matches PNG filenames with labels
        preprocess ('torchvision.transforms._presets.ImageClassification'>) : preprocessing transforms provided with the pretrained models
        """
        self.all_img_files = np.array([a for a in os.listdir(imgpath) if '.png' in a])
        self.imgpath = imgpath   
        self.preprocess = preprocess  
   

    def __getitem__(self, index):     
        img = decode_image(os.path.join(self.imgpath,  self.all_img_files[index]))  
        # Apply inference preprocessing transforms
        if self.preprocess is not None:
            img = self.preprocess(img) # .unsqueeze(0)

        filename = self.all_img_files[index]
        return (img, filename)
    
    def __len__(self):
        return (len(self.all_img_files))
    

class FeatureExtractor:
    """
    """

    def __init__(self, model_tag):
        """
        """
        self.model_tag = model_tag
        self.model, weights = self._load_pretraind_model(model_tag)
        self.preprocessor = weights.transforms()
        self.train_nodes, self.eval_nodes = get_graph_node_names(self.model)


    def _load_pretraind_model(self, model_tag):
        """
        """
        if model_tag == "ResNet50":
            from torchvision.models import resnet50, ResNet50_Weights
            weights = ResNet50_Weights.IMAGENET1K_V2
            model = resnet50(weights=weights)
        elif model_tag == "DenseNet121":
            from torchvision.models import densenet121, DenseNet121_Weights 
            weights = DenseNet121_Weights.IMAGENET1K_V1
            model = densenet121(weights=weights)
        elif model_tag == "MobileNet_V3_Large":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            model = mobilenet_v3_large(weights=weights)
        elif model_tag == "MobileNet_randinit":
            from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
            model = mobilenet_v3_large(weights=None)
        elif model_tag == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            weights = VGG16_Weights.IMAGENET1K_V1
            model = vgg16(weights=weights)
        # Transformers 
        elif model_tag == "Vit_b_16":
            from torchvision.models import vit_b_16, ViT_B_16_Weights
            weights = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
            model = vit_b_16(weights=weights)
        elif model_tag == "MaxVit_T":
            from torchvision.models import maxvit_t, MaxVit_T_Weights  
            weights = MaxVit_T_Weights.IMAGENET1K_V1     
            model = maxvit_t(weights=weights)  
        elif model_tag == "Swin_S":
            from torchvision.models import swin_s, Swin_S_Weights
            weights = Swin_S_Weights.IMAGENET1K_V1
            model = swin_s(weights=weights)
        else:
            print("not a valid model_tag")
        return(model, weights)  


    def _dim_reduce(self, X, n_neigh, n_dims_red):
        """
        """
        scaler = StandardScaler()
        reducer = umap.UMAP(
            n_neighbors = n_neigh, 
            n_components = n_dims_red, 
            metric = 'euclidean',
            n_jobs = -1
            )
        X_scaled = scaler.fit_transform(X)
        X_trans = reducer.fit_transform(X_scaled)
        X_out = scaler.fit_transform(X_trans)
        return(X_out)
    

    def create(self, fex_tag):  
        """
        """   
        return_nodes = {fex_tag: "feature_1"}
        self.extractor = create_feature_extractor(self.model, return_nodes=return_nodes)
        self.fex_tag = fex_tag
        _ = self.extractor.eval()


    def extract(self, image_path, freq_pool, batch_size, n_batches = 2):
        """
        """
        dataset = ImageDataset(image_path, self.preprocessor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=False, drop_last=False)
        self.X_li = [] # features
        self.N_li = [] # file Nanes
        self.X = []
        self.N = []
        self.featu_path = os.path.dirname(image_path)
        for ii, (batch, finam) in enumerate(loader, 0):
            print('Model:', self.model_tag )
            print('Feature layer:', self.fex_tag )
            print('Input resized image:', batch.shape)
            # batch = batch.to(torch.float)
            pred = self.extractor(batch)['feature_1'].detach().numpy() 
            print('Feature out of net:', pred.shape)
            # blockwise pooling along frequency axe 
            pred = skimage.measure.block_reduce(pred, (1,1,freq_pool,1), np.mean)
            print('After average pool along freq:', pred.shape)

            # cutting time edges (currently hard coded to 20% on each side)
            ecut = np.ceil(0.20 * pred.shape[3]).astype(int)
            # print('ecut', ecut)
            pred = pred[:, :, :, ecut:(-1*ecut)] 
            print('NEW - After cutting time edges:', pred.shape)
            
            # full average pool over time (do asap to avoid memory issues later)
            pred = pred.mean(axis=3)
            print('After average pool along time:', pred.shape)
            # unwrap freq int feature dim
            pred = np.reshape(pred, shape=(pred.shape[0], pred.shape[1]*pred.shape[2]))
            print('After reshape:', pred.shape)
            print("")
            # do it dirty
            self.X_li.append(pred)
            self.N_li.append(np.array(finam))
            # dev
            if ii > n_batches:
                break   
        self.X = np.concatenate(self.X_li)
        self.N = np.concatenate(self.N_li)
        # save the full 'array shaped' features as npz with a ID timestamp
        tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        self.out_name = os.path.join(self.featu_path, tstmp + 'full_features_' + self.model_tag + '_' + self.fex_tag + '.npz')
        np.savez(file = self.out_name, X = self.X, N = self.N)   


    def save_full_features(self):
        """ 
        """
        # handle if training was killed early
        self.X = np.concatenate(self.X_li)
        self.N = np.concatenate(self.N_li)
        tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        self.out_name = os.path.join(self.featu_path, tstmp + 'full_features_' + self.model_tag + '_' + self.fex_tag + '.npz')
        np.savez(file = self.out_name, X = self.X, N = self.N)   


    def reduce_dimension(self, npzfile_full_path = None, n_neigh = 10, reduced_dim = 8):
        """
        """
        # take in-class path to npz file if none provided via arguments 
        if npzfile_full_path == None:
            npzfile_full_path = self.out_name
        # separate path and file name
        file_name_in = os.path.basename(npzfile_full_path)  
        featu_path = os.path.dirname(npzfile_full_path)   
        # process 
        npzfile = np.load(npzfile_full_path)
        X = npzfile['X']
        N = npzfile['N']
        # make 2d feats needed for plot 
        self.X_2D  = self._dim_reduce(X, n_neigh, 2)
        self.X_red = self._dim_reduce(X, n_neigh, reduced_dim)
        # save as npz
        tag_dim_red = "dimred_" + str(reduced_dim) + "_neigh_" + str(n_neigh) + "_"
        file_name_out = '_'.join(file_name_in.split('_')[0:2]) + '_' + tag_dim_red + '_'.join(file_name_in.split('_')[4:])
        self.out_name_reduced = os.path.join(featu_path, file_name_out)
        np.savez(file = self.out_name_reduced, X_red = self.X_red, X_2D = self.X_2D, N = N)


    def plot_full_features(self, npzfile_full_path = None, n=50):  
        """
        """ 
        if npzfile_full_path == None:
            npzfile_full_path = self.out_name
        npzfile = np.load(os.path.join(npzfile_full_path))
        X = npzfile['X']
        XS, _ = train_test_split(X, train_size=n, random_state=6666, shuffle=True)
        return(px.scatter(data_frame = XS.T, title = npzfile_full_path))     
    

    def plot_reduced_features(self, npzfile_reduced_path = None, n=50):  
        """
        """ 
        if npzfile_reduced_path == None:
            npzfile_reduced_path = self.out_name_reduced
        npzfile = np.load(os.path.join(npzfile_reduced_path))
        X = npzfile['X_red']
        XS, _ = train_test_split(X, train_size=n, random_state=6666, shuffle=True)
        return(px.scatter(data_frame = XS.T, title = npzfile_reduced_path))     

