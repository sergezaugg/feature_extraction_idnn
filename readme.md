
# Extract features from spectrograms with pre-trained image DNNs and dim-reduction 

### Overview 
* aaa
* ffffff
* aaaaaa

### Source data
* Acoustic recordings are from [xeno-canto](https://xeno-canto.org/)
* Standardized acoustic data preparation was performed with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)  
* In a nutshell: MP3 converted to WAVE, resampled, transformed to spectrograms, and stored as RGB images.
* Images can then be fed into the backend code for feature extractions

### Backend code
* Backend code is for feature extraction and dim-reduction 
* Code is in this subdir ```./pt_extract_features```
* Main functionality is in this module: ```utils_ml```
* Short example in ```main.py```
* There are 3 flat scripts used to perform the extraction: ```01_ 02_ 03_```
* Full and reduced-dim features as stored as NPZ files
* NPZ file are then stored on a Kaggle dataset [example](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01) where the frontend will fetch them.

### Dependencies / Intallation
* Developed under Python 3.12.8
* Make a fresh venv!
```bash 
pip install -r requirements.txt
```
* For feature extraction you also need to install **torch** and **torchvision**
* This code was developed under Windows with CUDA 12.6 and Python 3.12.8 
```bash 
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
* If other CUDA version needed, check instructions here https://pytorch.org/get-started/locally


