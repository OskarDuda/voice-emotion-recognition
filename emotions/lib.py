from datetime import datetime
import os
import sys
import time

import numpy as np
import pandas as pd
import pickle
from soundfile import read as wav_read
from sklearn.model_selection import RandomizedSearchCV
import xgboost


def timeit(f):
    def wrapper(*args, **kwargs):
        start = time.time()
        print("\t{} started".format(f.__name__))
        result = f(*args, **kwargs)
        stop = time.time()
        print("\t{} finished after {:.3f} s".format(f.__name__, stop - start))
        return result

    return wrapper


@timeit
def get_spectrum(signal):
    n = len(signal)
    print("\tSignal length: {}".format(n))
    sig_fft = np.fft.fft(signal)
    abs = np.abs(sig_fft)
    output = abs[:n // 2]
    return output


def normalize_signal_length(signal, n):
    if len(signal) >= n:
        output = signal[:n]
        print("\tSignal cut from {} to {} samples".format(len(signal), len(output)))
        return output
    else:
        deficit = n - len(signal)
        output = np.append(signal, np.zeros(deficit))
        print("\tSignal extended from {} samples to {}".format(len(signal), len(output)))
        return output


@timeit
def aggregate_signal_by_parts(signal, n=10, fun=np.average):
    part_size = len(signal) // n
    split_signal = []
    for i in range(n):
        split_signal.append(fun(signal[part_size * i:part_size * (i + 1)]))
    return split_signal


@timeit
def count_data(folders):
    n = 0
    for g in folders:
        for actor in os.listdir(os.path.join(g)):
            for file in os.listdir(os.path.join(g, actor)):
                n += 1
    return n


def preprocess_data(folders, n_features):
    X_cols = ['feat_{}'.format(k) for k in range(1, n_features + 1)]
    df = pd.DataFrame(dict(zip(X_cols, [[] for k in X_cols])))
    df['y'] = []
    data_n = count_data(folders)
    i = 0
    for g in folders:
        for actor in os.listdir(os.path.join(g)):
            for file in os.listdir(os.path.join(g, actor)):
                i += 1
                try:
                    print("{:.1f}% \n\tFile: {}".format(i / data_n * 100, file))
                    path = os.path.join(g, actor, file)
                    v = wav_read(path)[0]
                    if len(v.shape) > 1:
                        v = v[:, 0]
                    v = normalize_signal_length(v, 600000)
                    X_row = pd.Series(dict(zip(X_cols, aggregate_signal_by_parts(get_spectrum(v), n_features))))
                    df.loc[i, X_cols] = X_row
                    df.loc[i, 'y'] = int(file.split('.')[0].split('-')[2])
                except Exception as e:
                    print('ERROR: {}\n\tSkipping signal'.format(e))
    return df


def load_cached_data(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def cache_data(data, path):
    with open(path, 'wb+') as f:
        pickle.dump(data, f)


@timeit
def get_data(n_features, folders, preprocessed_folder, always_preprocess_data=False,
             cache_new_data=False, file_path=None):
    preprocessed_data_exists = any(os.listdir(preprocessed_folder))
    data_should_be_loaded = preprocessed_data_exists and not always_preprocess_data
    if data_should_be_loaded:
        df = load_cached_data(file_path)
    else:
        df = preprocess_data(folders, n_features)
        if cache_new_data:
            cache_data(df, file_path)
    return df


def fun_call_by_name(val):
    if '.' in val:
        module_name, fun_name = val.rsplit('.', 1)
        assert module_name.startswith('scipy')
    else:
        module_name = 'emotions.lib'
        fun_name = val
    __import__(module_name)
    module = sys.modules[module_name]
    fun = getattr(module, fun_name)
    return fun


def parse_random_search_params(**params):
    '''
    Parses randomized search parameters taken from configurations yaml. List parameters are passed over unchanged, dictionary
    parameters are replace with corresponding functions.
    '''
    dist_params = params['param_distributions']
    for key in dist_params.keys():
        if isinstance(dist_params[key], list):
            pass
        elif isinstance(dist_params[key], dict):
            params['param_distributions'][key] = fun_call_by_name(dist_params[key]['function_name'])(
                                                                  **dist_params[key]['parameters'])
    return params


def build_new_model(X_train, y_train, default_model_params, search_params):
    clf = xgboost.XGBClassifier(**default_model_params)
    searcher = RandomizedSearchCV(clf, **search_params)
    searcher.fit(X_train, y_train)
    return searcher.best_estimator_


def get_model(X_train, y_train, configs):
    model_path = configs['model_path']
    search_params = parse_random_search_params(**configs['randomized_search_params'])
    default_model_params = configs['default_model_params']
    cache_new_model = configs['flags']['cache_new_model']
    always_build_new_model = configs['flags']['always_build_new_model']

    cached_model_exists = os.path.isfile(model_path)
    model_should_be_built = not cached_model_exists or always_build_new_model
    if model_should_be_built:
        model = build_new_model(X_train, y_train, default_model_params, search_params)
        if cache_new_model:
            base_path, extension = model_path.split('.')
            output_path = base_path+datetime.now().strftime('%y_%m_%d_%H_%M_%S')+'.'+extension
            cache_data(model, output_path)
    else:
        model = load_cached_data(model_path)
    return model


if __name__ == '__main__':
    import yaml
    configs_file = 'default_configs.yml'
    with open(configs_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)

    p = parse_random_search_params(**configs['randomized_search_params'])