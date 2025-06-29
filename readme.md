
# Feature extraction from spectrograms with pre-trained image DNNs (fe_idnn)

### Overview
* Primarily developed for acoustic recordings from [xeno-canto](https://xeno-canto.org/)
* Download of mp3 and conversion to spectrograms as RGB images can be performed with [this tool](https://github.com/sergezaugg/xeno_canto_organizer)  
* Images can then be processed with the **IDNN_extractor** class provided here
* First, extract array features from inner layers of pre-trained image DNNs and transform to long linear features
* Second, dim reduce the linear features with UMAP to shorter feature vectors
* Full and reduced-dim features as stored as NPZ files

### Usage
* Main functionality called via a single class **IDNN_extractor**
* Illustration with short a script [sample_code_mini.py](sample_code_mini.py)
* Examples to extract from several layers here [sample_code_full.py](sample_code_full.py)

### Companion project
* NPZ files can then stored on a Kaggle dataset [(example)](https://www.kaggle.com/datasets/sezaugg/spectrogram-clustering-01) where the [frontend](https://spectrogram-image-clustering.streamlit.app/) will fetch them [(Github repo)](https://github.com/sergezaugg/spectrogram_image_clustering)  

### Intallation (usage in Python project)
* Make a fresh venv an install fe_idnn from Python package wheel found on [this github repo](https://github.com/sergezaugg/feature_extraction_idnn/releases)
* ```pip install https://github.com/sergezaugg/feature_extraction_idnn/releases/download/vx.x.x/fe_idnn-x.x.x-py3-none-any.whl```
* **torch** and **torchvision** must be installed separately for specific CUDA version
* ```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126``` (e.g. for Windows with CUDA 12.6 and Python 3.12.8)
* If other CUDA version needed, check instructions here https://pytorch.org/get-started/locally

### Intallation (usage in Kaggle notebook)
* On Kaggle packages are already installed, for notebooks try this:
* ```!pip install --no-deps https://github.com/sergezaugg/feature_extraction_idnn/releases/download/vx.x.x/fe_idnn-x.x.x-py3-none-any.whl```

### Intallation (development)
* Clone the repo and navigate to its root dir
* ```pip install -r requirements.txt```
* ```pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126``` (for Windows with CUDA 12.6 and Python 3.12.8)
* If other CUDA version, check instructions here https://pytorch.org/get-started/locally

### External dependencies
* Instantiation of IDNN_extractor will trigger download of pre-trained models from https://download.pytorch.org/models

### Project Structure
```
├── dev/                # data, dirs and scripts for devel
├── pics/               # Images for documentation
├── src/                # Source code (Python package) and unit tests
├── pyproject.toml      # Build configuration
├── requirements.txt    # Python dependencies
├── sample_code_full.py # Example usage script
└── sample_code_mini.py # Example usage script
```

### ML details
<img src="pics/spectro_imDNN_data_flow.png" alt="Example image" width="600"/>

---

