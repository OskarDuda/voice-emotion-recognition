import click
import lib
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yaml


@click.command()
@click.option('--configs_file', '-c', default='default_configs.yml')
def main(configs_file):
    with open(configs_file, 'r') as f:
        configs = yaml.load(f)
    data_folders = configs['data_dirs']
    n_features = int(np.sqrt(lib.count_data(data_folders)))
    scaler = MinMaxScaler()
    emotion_id = configs['emotion_ids_all'].get(configs['emotion_to_predict'])
    df = lib.get_data(n_features, data_folders,
                      preprocessed_folder=configs.get('cached_data_dir'),
                      always_preprocess_data=configs['flags'].get('always_preprocess_data'),
                      cache_new_data=configs['flags'].get('cache_preprocessed_data'),
                      file_path=configs.get('cached_data_path'))
    X = df.drop('y', axis=1)
    X = scaler.fit_transform(X)
    y = (df['y'] == emotion_id).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # if cached_data_filename in os.listdir():
    #     with open('data_preprocessed.pkl', 'rb') as f:
    #         df = pickle.load(f)
    # else:
    #     df = load_data(n_features)
    #     with open('data_preprocessed.pkl', 'wb+') as f:
    #         pickle.dump(df, f)

    # clf = xgboost.XGBClassifier(**configs['default_model_params'])
    model = lib.get_model(X_train,
                          y_train,
                          configs['model_path'],
                          configs['randomized_search_params'],
                          configs['default_model_params'],
                          configs['flags']['cache_new_model'],
                          configs['flags']['always_build_new_model'])
    print("Fitting the classifier")
    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)
    print("AUC: {:.3f}%".format(score * 100))


if __name__ == "__main__":
    main()
    # directory = os.path.join('Data', 'Audio_Song_Actors_01-24', 'Actor_01')
    # filename = '03-02-01-01-01-01-01.wav'
    # file_path = os.path.join(directory, filename)
    #
    # v2 = wav_read(file_path)[1]
    # fig, axes = plt.subplots(2,1)
    # axes[0].plot(get_spectrum(v2))
    # axes[1].plot(aggregate_signal_by_parts(get_spectrum(v2), 30))
