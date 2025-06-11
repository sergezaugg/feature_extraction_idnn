
# Extract features from spectrograms with pre-trained image DNNs and dim-reduction 

### Overview

* Intended for acoustic recordings from [xeno-canto](https://xeno-canto.org/)
* Download and acoustic data preparation can be performed with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)  
    * In a nutshell: MP3 converted to WAVE, resampled, segmented, transformed to spectrograms, and stored as RGB images.
* Images can then be processed with the **FeatureExtractor** Class provided here
    * First, extract array features from inner layers of pre-trained image DNNs
    * Second, transform arrays to rather long linear features
    * Third, dim reduce the linear features with UMAP to shorter feature vectors
    * Full and reduced-dim features as stored as NPZ files
* NPZ files can then stored on a Kaggle dataset [(example)](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01) where the [frontend](https://spectrogram-image-clustering.streamlit.app/) will fetch them.
*  [Link to frontend](https://spectrogram-image-clustering.streamlit.app/) and its [Github repo](https://github.com/sergezaugg/spectrogram_image_clustering)  

### Usage
* Code is in this subdir ```./pt_extract_features```
* Main functionality called via a single class **FeatureExtractor** defined is in ```utils_ml.py```
* Short example: ```main.py```
* Full example: ```history.py```

### Dependencies / Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
* ```pip install -r requirements.txt```
* **torch** and **torchvision** must also be installed
* This code was developed under Windows with CUDA 12.6 and Python 3.12.8 
* ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126```
* If other CUDA version needed, check instructions here https://pytorch.org/get-started/locally

### ML details
![](pics/spectro_imDNN_data_flow.png)

