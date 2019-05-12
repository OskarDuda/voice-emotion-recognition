# Voice emotion recognition

## How to use
```bash
python emotions/main.py --configs_file path/to/my_configs.yml
``` 

## How to construct configs YAML file
A default configuration file has been attached as a reference. All necessary keys explained below:
 
 - data_dirs - folders containing voice wav files
 - cached_data_path - (optional) full path to cached, preprocessed data with features extracted saved as .pkl 
 - flags - all bolean flags necessary for workflow control
   - always_build_new_model - a new model is built every time app is run if set to True, uses cached model otherwise
   - cache_new_model - saves a new model after building it if set to True
   - always_preprocess_data - preprocesses data and extracts features every time app is run if set to True, uses cached data otherwise
   - cache_preprocessed_data - saves preprocess data with features extracted if set to True
 - randomized_search_params - contains parameters for randomized CV search, more details on scikit-learn website [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
 - default_model_params - contains constant parameters for model, that won't be optimized using randomized search
 - emotion_ids_all - all emotions available and IDs corresponding to them, the emotion ID needs to be included in .wav files' names as described [here](https://zenodo.org/record/1188976#.XNiRgkOxWWi)
 - emotion_to_predict - name of emotion to be predicted, needs to be a key from 'emotion_ids_all'     
 