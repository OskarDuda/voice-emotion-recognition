import click
from emotions import lib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml


@click.command()
@click.option('--configs_file', '-c', default='default_configs.yml')
def main(configs_file):
    with open(configs_file, 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    data_folders = configs['data_dirs']
    n_features = int(np.sqrt(lib.count_data(data_folders)))
    scaler = MinMaxScaler()
    emotion_id = configs['emotion_ids_all'].get(configs['emotion_to_predict'])
    df = lib.get_data(n_features, data_folders,
                      preprocessed_folder=configs.get('cached_data_dir'),
                      always_preprocess_data=configs['flags'].get('always_preprocess_data'),
                      cache_new_data=configs['flags'].get('cache_preprocessed_data'),
                      file_path=configs.get('cached_data_path'))
    df = df.sample(frac=1)
    X = df.drop('y', axis=1)
    X = scaler.fit_transform(X)
    y = (df['y'] == emotion_id).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = lib.get_model(X_train, y_train, configs)
    print("Fitting the classifier")
    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)
    print("AUC: {:.3f}%".format(score * 100))


if __name__ == "__main__":
    main()
