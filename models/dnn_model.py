import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
import shutil  #used for deleting model in temp directory once predictions are made
import keras
from keras import layers
from keras import models
from keras import regularizers
import json
import os
from models import df_utilities as dfu


# if you specify a test dataset, you probably want only test returned.
def normalize(train_pandas, test_pandas):
    mean = train_pandas.mean(axis=0)
    train_pandas -= mean
    std = train_pandas.std(axis=0)
    train_pandas /= std
    train_pandas = train_pandas.dropna(axis=1)
    if test_pandas is not None:
        test_pandas -= mean
        test_pandas /= std
        test_pandas = test_pandas.dropna(axis=1)
        return test_pandas
    else:
        return train_pandas


def trim_correlated(train_pandas, test_pandas, threshold):
    df_corr = train_pandas.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = train_pandas[un_corr_idx]
    if test_pandas is not None:
        df_test_out = test_pandas[un_corr_idx]
        return df_out, df_test_out
    else:
        return df_out


# this has to be done because I can't pickle the tensorflow model files, can only do so for the compressed model files
class ModelFiles:
    def __init__(self, path):
        self.assetfolder_string = "assets"
        self.variablefolder_string = "variables"
        self.saved_model_filename = "saved_model.pb"
        self.variables_index_filename = "variables.index"
        self.variables_DATA_00000_OF_00001_filename = "variables.DATA-00000-OF-00001"
        self.saved_model_pb = None
        self.variables_index = None
        self.variables_DATA_00000_OF_00001 = None
        self.destination_path = None
        self.destination_assets_subpath = None
        self.destination_variables_subpath = None
        self.saved_model_pb = self.openandclose(path, "", self.saved_model_filename)
        self.variables_index = self.openandclose(path, self.variablefolder_string, self.variables_index_filename)
        self.variables_DATA_00000_OF_00001 = self.openandclose(path, self.variablefolder_string,
                                                               self.variables_DATA_00000_OF_00001_filename)

    def openandclose(self, path, subfolder, filename):
        f = open(path + "/" + subfolder + "/" + filename, "rb")
        file_content = f.read()
        f.close()
        return file_content

    def create_model_directory_from_binary(self, parent_dir, modelnum):
        modelnumdir_string = "model" + str(modelnum)
        self.destination_path = parent_dir + "/" +modelnumdir_string
        if not os.path.exists(self.destination_path):
            os.makedirs(self.destination_path)
            self.destination_assets_subpath = os.path.join(self.destination_path, self.assetfolder_string)
            self.destination_variables_subpath = os.path.join(self.destination_path, self.variablefolder_string)
            os.mkdir(self.destination_assets_subpath)
            os.mkdir(self.destination_variables_subpath)
            # this step actually replicates all the files in the correct places
            self.populate_model_directory_from_binary()
        return self

    def populate_model_directory_from_binary(self):
        pbFile = open(self.destination_path + "/" + self.saved_model_filename, "wb")
        pbFile.write(self.saved_model_pb)
        index_File = open(self.destination_variables_subpath + "/" + self.variables_index_filename, "wb")
        index_File.write(self.variables_index)
        DATA_File = open(self.destination_variables_subpath + "/" + self.variables_DATA_00000_OF_00001_filename, "wb")
        DATA_File.write(self.variables_DATA_00000_OF_00001)

class Model:
    def __init__(self, df_training, remove_log_p_descriptors):
        self.model_files_objects = None
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.df_training = df_training
        self.is_binary = None
        self.temp_path = os.path.join(os.getcwd(), "temp")
        self.epochs = 150
        self.k = 5
        self.version = "1.7"
        self.remove_corr = True
        self.corr_threshold = 0.95
        self.un_corr_idx = None
        self.qsar_method = 'Deep Neural Network'
        self.description = 'keras implementation of DNN'

    def build_model(self):
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            dfu.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, self.remove_corr)
        self.descriptor_names = train_column_names
        train_features_pandas = pd.DataFrame(train_features)
        self.un_corr_idx = train_features_pandas.index
        normalized_features_pandas = normalize(train_features_pandas, None)
        normalized_features_np = np.array(normalized_features_pandas)
        self.kxfoldvalidation(normalized_features_np, train_labels)
        return self


    def kerasmodel(self, train_data_np, test_data_np, index):
        model = models.Sequential()
        model.add(layers.Dense(144, activation='relu', kernel_regularizer=regularizers.l1(0.003),
                               input_shape=(train_data_np.shape[1],))) # 144
        # model.add(layers.Dropout(0.5))
        # model.add(BatchNormalization())
        model.add(layers.Dense(81, kernel_regularizer=regularizers.l1(0.003), activation='relu')) # 81
        model.add(layers.Dense(36, activation='relu'))
        model.add(layers.Dense(18, activation='relu'))
        model.add(layers.Dense(9, activation='relu'))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        model.fit(train_data_np, test_data_np, epochs=self.epochs, batch_size=1, verbose=0)
        model.save(self.temp_path + "/model" + str(index))

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)


        tf.keras.backend.clear_session()
        return model

    def kxfoldvalidation(self, features_np, targets_np):
        k = self.k
        num_validation_samples = len(features_np) // k
        all_train_mae = []
        all_train_mse = []
        all_scores_mse = []
        all_scores_mae = []

        all_scores_loss = []
        all_scores_accuracy = []
        all_train_loss = []
        all_train_accuracy = []

        if not os.path.exists(self.temp_path):
            os.mkdir(self.temp_path)

        self.model_files_objects = list()

        for i in range(k):
            print('processing fold #', i)
            val_data = features_np[i * num_validation_samples: (i + 1) * num_validation_samples]
            val_targets = targets_np[i * num_validation_samples: (i + 1) * num_validation_samples]

            partial_train_data = np.concatenate(
                [features_np[:i * num_validation_samples], features_np[(i + 1) * num_validation_samples:]], axis=0)
            partial_train_targets = np.concatenate(
                [targets_np[:i * num_validation_samples],
                 targets_np[(i + 1) * num_validation_samples:]], axis=0)
            if self.is_binary == False:
                model = self.kerasmodel(partial_train_data, partial_train_targets, i)
                train_mse, train_mae = model.evaluate(partial_train_data, partial_train_targets, verbose=0)
                val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

                all_train_mse.append(train_mse)
                all_train_mae.append(train_mae)
                all_scores_mse.append(val_mse)
                all_scores_mae.append(val_mae)

                model_files_object = ModelFiles(self.temp_path + "/model" + str(i))
                self.model_files_objects.append(model_files_object)


            else:
                model = self.categorical_model(partial_train_data, partial_train_targets, i)
                val_loss, val_accuracy = model.evaluate(val_data, val_targets, verbose=0)
                all_scores_accuracy.append(val_accuracy)

                model_files_object = ModelFiles(self.temp_path + "/" + "model" + str(i))
                self.model_files_objects.append(model_files_object)


        print("all scores mse =", all_scores_mse)
        print("all scores mae =", all_scores_mae)
        print("train scores mse =", all_train_mse)
        print("train scores mae =", all_train_mae)

        print("all scores accuracy", all_scores_accuracy)


    def categorical_model(self, train_data_np, test_data_np, index):
        model = models.Sequential()
        model.add(layers.Dense(400, activation='relu', kernel_regularizer=regularizers.l1(0.001),
                               input_shape=(train_data_np.shape[1],)))
        model.add(layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l1(0.001)))
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data_np, test_data_np, epochs=self.epochs, batch_size=1, verbose=0)
        model.save(self.temp_path + "/model" + str(index))
        tf.keras.backend.clear_session() # I'm not sure this does anything
        return model

    def do_predictions(self, df_prediction):
        train_ids, train_labels, train_features, train_column_names, isbinary = \
            dfu.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, self.remove_corr)
        pred_ids, pred_labels, pred_features = dfu.prepare_prediction_instances(df_prediction, train_column_names)
        train_features_pandas = pd.DataFrame(train_features)
        pred_features_pandas = pd.DataFrame(pred_features)
        normalized_test_features_pandas = normalize(train_features_pandas, pred_features_pandas)
        normalized_test_features_np = np.array(normalized_test_features_pandas)

        self.create_models_from_binary(self.temp_path)

        predictionslist = []
        for i in range(self.k):
            model_loaded = keras.models.load_model(self.temp_path + "/model" + str(i))
            predictions = model_loaded.predict(normalized_test_features_np)
            predictionslist.append(predictions)
        prediction_np = np.array(predictionslist)
        avg_predictions = np.mean(prediction_np, axis=0)
        if self.is_binary is not True:
            r2 = r2_score(pred_labels, avg_predictions)
            print('Rsq for Test data = ', r2)
        else:
            avg_predictions_binary = np.where(avg_predictions > 0.5, 1, 0)
            BA = balanced_accuracy_score(pred_labels, avg_predictions_binary)
            print('Balanced Accuracy for Test data =', BA)

        shutil.rmtree(self.temp_path)

        return avg_predictions

    def create_models_from_binary(self, model_dropoff_path):
        for i in range(self.k):
            Modelfiles_i = self.model_files_objects[i]
            Modelfiles_i.create_model_directory_from_binary(model_dropoff_path, i)


class ModelDescription:
    def __init__(self, Model):
        self.is_binary = Model.is_binary
        self.version = Model.version
        self.remove_log_p_descriptors = Model.remove_log_p_descriptors
        self.version = Model.version
        self.qsar_method = Model.qsar_method
        self.description = Model.description
        self.corr_threshold = Model.corr_threshold
        self.n_folds = Model.k
        self.epochs = Model.epochs


    def to_json(self):
        return json.dumps(self.__dict__)


def main():
    df_training = pd.read_csv("LogBCF OPERA T.E.S.T. 5.1 training.tsv", delimiter='\t')
    df_prediction = pd.read_csv("LogBCF OPERA T.E.S.T. 5.1 prediction.tsv", delimiter='\t')
    # df_training = pd.read_csv("E:\datasets_benchmark_toxicity\Mutagenicity\Mutagenicity_prediction_set-2d.csv", quotechar='"')
    # df_prediction = pd.read_csv("E:\datasets_benchmark_toxicity\Mutagenicity\Mutagenicity_training_set-2d.csv", quotechar='"')
    remove_log_p_descriptors = False
    model = Model(df_training, remove_log_p_descriptors)
    model.build_model()
    predictions = model.do_predictions(df_prediction)
    print(ModelDescription(model).to_json())


if __name__ == "__main__":
    main()
    # path = "C:/Users/Weeb/Documents/python-qsar-ws/models/model0"
    # Modelfile0 = ModelFiles(path)
    # dropoffpath = "E:\models"
    # for i in range(5):
    #    Modelfile0.create_model_directory_from_binary(dropoffpath, i)
