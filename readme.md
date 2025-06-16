
# Extract features from spectrograms with pre-trained image DNNs and dim-reduction 

### Table of Contents
* [Overview](#Overview)
* [Intallation](#Intallation)
* [Usage](#Usage)
* [ML details](#ML-details)

### Overview
* Intended for acoustic recordings from [xeno-canto](https://xeno-canto.org/)
* Download of mp3 and conversion to spectrograms as RGB images can be performed with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)  
* Images can then be processed with the **FeatureExtractor** Class provided in [pt_extract_features/utils_ml.py](pt_extract_features/utils_ml.py)  pt_extract_features/utils_ml.py
* First, extract array features from inner layers of pre-trained image DNNs
* Second, transform arrays to rather long linear features
* Third, dim reduce the linear features with UMAP to shorter feature vectors
* Full and reduced-dim features as stored as NPZ files
* NPZ files can then stored on a Kaggle dataset [(example)](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01) where the [frontend](https://spectrogram-image-clustering.streamlit.app/) will fetch them.
*  [Link to frontend](https://spectrogram-image-clustering.streamlit.app/) and its [Github repo](https://github.com/sergezaugg/spectrogram_image_clustering)  

### Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
* ```pip install -r requirements.txt```
* **torch** and **torchvision** must also be installed
* This code was developed under Windows with CUDA 12.6 and Python 3.12.8 
* ```pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126```
* If other CUDA version needed, check instructions here https://pytorch.org/get-started/locally

### Usage
* Main functionality called via a single class **FeatureExtractor** defined is in ```utils_ml.py```
* Illustration with short interactive script here: ```main_mini.py```
* Batch process with CLI tool, example: ```python main_batch.py -i "./dev_data/images" -n 31```

### ML details
<img src="pics/spectro_imDNN_data_flow.png" alt="Example image" width="600"/>
