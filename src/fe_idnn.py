#--------------------             
# Author : Serge Zaugg
# Description : Functions an classes specific to ML/PyTorch backend
#--------------------

import os
import pandas as pd
import numpy as np
import torch
# import datetime
from torch.utils.data import Dataset
from torchvision.io import decode_image
from sklearn.preprocessing import StandardScaler
import umap.umap_ as umap
import skimage.measure
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.feature_extraction import get_graph_node_names


class ImageDataset(Dataset):
    """
    This PyTorch dataset loads PNG images from a specified directory, applies given preprocessing
    transforms, and returns the processed image along with its filename. It is designed
    to be used with PyTorch's DataLoader for batching and shuffling.
    Attributes:
        all_img_files (np.ndarray): Array of PNG image filenames in the provided directory.
        imgpath (str): Path to the directory containing PNG images.
        preprocess (callable): Preprocessing transforms (e.g., torchvision presets) to apply to the images.
    """
    
    def __init__(self, imgpath, preprocess):
        """
        Initializes the ImageDataset.
        Args:
            imgpath (str): Path to the directory containing PNG images.
            preprocess (callable): Preprocessing transforms to apply to the images.
                Typically, a torchvision.transforms.
        """
        self.all_img_files = np.array([a for a in os.listdir(imgpath) if '.png' in a])
        self.imgpath = imgpath   
        self.preprocess = preprocess  
   
    def __getitem__(self, index):
        """
        Retrieves and preprocesses an image at the specified index.
        Args:
            index (int): Index of the image to retrieve.
        Returns:
            tuple:
                img (Tensor): The preprocessed image tensor.
                filename (str): The filename of the image.
        """     
        img = decode_image(os.path.join(self.imgpath,  self.all_img_files[index]))  
        # Apply inference preprocessing transforms
        if self.preprocess is not None:
            img = self.preprocess(img) # .unsqueeze(0)

        filename = self.all_img_files[index]
        return (img, filename)
    
    def __len__(self):
        """
        Returns the total number of images in the dataset.
        """
        return (len(self.all_img_files))
    

class IDNN_extractor:
    """
    A class for extracting, saving, and visualizing deep features from images using pretrained models.
    This class provides methods to initialize various pretrained models, extract features from images,
    perform dimensionality reduction, and visualize both the full and reduced feature sets.
    """

    def __init__(self, model_tag):
        """
        Initializes the IDNN_extractor with the specified pretrained model.

        Args:
            model_tag (str): Tag specifying which pretrained model to use.
                Supported tags include "ResNet50", "DenseNet121", "MobileNet_V3_Large",
                "MobileNet_randinit", "vgg16", "Vit_b_16", "MaxVit_T", "Swin_S".
        """
        self.model_tag = model_tag
        self.model, weights = self._load_pretraind_model(model_tag)
        self.preprocessor = weights.transforms()
        self.train_nodes, self.eval_nodes = get_graph_node_names(self.model)


    def _load_pretraind_model(self, model_tag):
        """
        Loads a specified pretrained model and its weights.
        Args:
            model_tag (str): Tag specifying which pretrained model to load.
        Returns:
            tuple: (model (torch.nn.Module), weights (torchvision.models.Weights)) 
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
        Performs dimensionality reduction on the input feature array using UMAP.
        Args:
            X (np.ndarray): The input feature array.
            n_neigh (int): Number of neighbors to use in UMAP.
            n_dims_red (int): Target dimensions for reduction.
        Returns:
            np.ndarray: The dimensionally reduced feature array.
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
        Creates a feature extractor for the specified node in the model, saved as attribute self.extractor
        Args:
            fex_tag (str): The name of the feature node (inner layer) from the model to use for feature extraction
        """   
        return_nodes = {fex_tag: "feature_1"}
        self.extractor = create_feature_extractor(self.model, return_nodes=return_nodes)
        self.fex_tag = fex_tag
        _ = self.extractor.eval()


    def extract(self, image_path, fe_save_path, freq_pool, batch_size, n_batches = 2, ecut = 0):
        """
        Extracts array-features from images in 'image_path', processes these array and applies pooling, and saves the features.
        Args:
            image_path (str): Path to directory containing images.
            freq_pool (int): Pooling window size along the frequency axis.
            batch_size (int): Number of images per batch.
            n_batches (int, optional): Number of batches to process (default is 2).
            ecut (int, >= 0) : nb bins to cut from time edges
        Side Effects:
            Saves the extracted features and corresponding filenames as a .npz file in the parent of image directory.
        """
        dataset = ImageDataset(image_path, self.preprocessor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,  shuffle=False, drop_last=False)
        self.X_li = [] # features
        self.N_li = [] # file Nanes
        self.X = []
        self.N = []
        # define where extracted features will be saved
        self.featu_path = fe_save_path
        # loop over images 
        for ii, (batch, finam) in enumerate(loader, 0):
            print('Model:', self.model_tag )
            print('Feature layer:', self.fex_tag )
            print('Input resized image:', batch.shape)
            # batch = batch.to(torch.float)
            pred = self.extractor(batch)['feature_1'].detach().numpy() 
            print('Feature out of net:           ', pred.shape)
            # blockwise pooling along frequency axe 
            pred = skimage.measure.block_reduce(pred, (1,1,freq_pool,1), np.mean)
            print('After average pool along freq:', pred.shape)
            # cutting time edges
            if ecut != 0:
                pred = pred[:, :, :, ecut:(-1*ecut)] 
            print('After cutting time edges:     ', pred.shape)
            # full average pool over time (do asap to avoid memory issues later)
            pred = pred.mean(axis=3)
            print('After average pool along time:', pred.shape)
            # unwrap freq int feature dim
            pred = np.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2]))
            print('After reshape:', pred.shape)
            print("")
            # do it dirty
            self.X_li.append(pred)
            self.N_li.append(np.array(finam))
            # dev
            if ii >= n_batches-1:
                break   
        self.X = np.concatenate(self.X_li)
        self.N = np.concatenate(self.N_li)
        # save the full 'array shaped' features as npz with a ID timestamp
        # tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        self.out_name = os.path.join(self.featu_path, 'full_features_' + self.model_tag + '_' + self.fex_tag + '.npz')
        np.savez(file = self.out_name, X = self.X, N = self.N)   


    def save_full_features(self):
        """ 
        Concatenates and saves the extracted features and filenames in .npz format.
        This method is meant to manually save features if extraction with .extract() was interrupted.
        """
        # handle if training was killed early
        self.X = np.concatenate(self.X_li)
        self.N = np.concatenate(self.N_li)
        # tstmp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_")
        self.out_name = os.path.join(self.featu_path, 'full_features_' + self.model_tag + '_' + self.fex_tag + '.npz')
        np.savez(file = self.out_name, X = self.X, N = self.N)   


    def reduce_dimension(self, n_neigh = 10, reduced_dim = 8):
        """
        Reduces the dimension of saved features using UMAP and saves the reduced features.
        Also always makes a 2d reduced version that is used for plotting.
        Args:
            npzfile_full_path (str, optional): Full path to the .npz file with full features.
            n_neigh (int, optional): Number of neighbors for UMAP (default is 10).
            reduced_dim (int, optional): Target dimension for reduction (default is 8).
        Side Effects:
            Saves the reduced-dim and 2D features in a .npz file.
        """
        # take in-class path to npz file 
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
        # file_name_out = '_'.join(file_name_in.split('_')[0:2]) + '_' + tag_dim_red + '_'.join(file_name_in.split('_')[4:])
        file_name_out = tag_dim_red + '_'.join(file_name_in.split('_')[2:])

        self.out_name_reduced = os.path.join(featu_path, file_name_out)
        np.savez(file = self.out_name_reduced, X_red = self.X_red, X_2D = self.X_2D, N = N)


 
    

  

