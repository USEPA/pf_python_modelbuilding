import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
import shutil  #used for deleting model in temp directory once predictions are made
import keras
from keras import layers
from keras import models
from keras import regularizers
import json
import os
import time
from models import df_utilities as DFU

def normalize(train_pandas, test_pandas):
    """ technically a redundant function that is handled in the DFU.prepareinstances method but I have noticed my models
    are much worse if I don't send train_features and pred_featuresthrough. Normalizes the data and returns test data
    based on the mean and standard deviation determined by the train data."""
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


class ModelFiles:
    """this has to be done because I can't pickle the tensorflow model files, can only do so for the compressed model
    files this class describes everything about a folder directory that tensorflow uses to store model file binaries
    about weights, biases, etc.unlike random forest where in the model class self.rfr equals whatever the model object
    is, I have to set its Model.model_files_objects equal to these classes, these classes are capable of picking up
    everything in tensorflow model folders, and also recreating those folders whenever a prediction is made"""
    def __init__(self, path):
        self.assetfolder_string = "assets"
        self.variablefolder_string = "variables"
        self.saved_model_filename = "saved_model.pb"
        self.variables_index_filename = "variables.index"
        self.variables_DATA_00000_OF_00001_filename = None
        self.saved_model_pb = None
        self.variables_index = None
        self.variables_DATA_00000_OF_00001 = None
        self.destination_path = None
        self.destination_assets_subpath = None
        self.destination_variables_subpath = None
        self.saved_model_pb = self.openandclose(path, "", self.saved_model_filename)
        self.variables_index = self.openandclose(path, self.variablefolder_string, self.variables_index_filename)

        if (os.name == "nt"):
            self.variables_DATA_00000_OF_00001_filename = "variables.DATA-00000-OF-00001"
        elif (os.name == "posix"):
            self.variables_DATA_00000_OF_00001_filename = "variables.data-00000-of-00001"

        self.variables_DATA_00000_OF_00001 = self.openandclose(path, self.variablefolder_string,
                                                               self.variables_DATA_00000_OF_00001_filename)

        # print("check", os.name, self.variables_DATA_00000_OF_00001_filename)


    def openandclose(self, path, subfolder, filename):
        """opens and closes files and saves the content in the ModelFiles object"""
        f = open(path + "/" + subfolder + "/" + filename, "rb")
        file_content = f.read()
        f.close()
        return file_content

    def create_model_directory_from_binary(self, parent_dir, modelnum):
        """this method is used whenever binaries for models are sent through the web service (like when only a
        prediction request is made) all the binaries are unpacked and tensorflow is able to use the folders it
        recognizes (like model0 - model4) to reconstruct the models"""

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
        """populates directories from binary of model file that gets sent through the web service for prediction"""
        pbFile = open(self.destination_path + "/" + self.saved_model_filename, "wb")
        pbFile.write(self.saved_model_pb)
        index_File = open(self.destination_variables_subpath + "/" + self.variables_index_filename, "wb")
        index_File.write(self.variables_index)
        DATA_File = open(self.destination_variables_subpath + "/" + self.variables_DATA_00000_OF_00001_filename, "wb")
        DATA_File.write(self.variables_DATA_00000_OF_00001)

class Model:
    """model class that houses most information about the model tuning."""
    def __init__(self, df_training, remove_log_p_descriptors):
        self.model_files_objects = None
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.df_training = df_training
        self.is_binary = None
        self.temp_path = os.path.join(os.getcwd(), "temp" + str(time.time()))
        self.epochs = 300
        self.batch = 32
        self.k = 5
        self.version = "1.8"
        self.remove_corr = True
        self.corr_threshold = 0.95
        self.un_corr_idx = None
        self.qsar_method = 'Deep Neural Network'
        self.description = 'keras implementation of DNN'

    def build_model(self):
        """ model building method that calls the k-fold cross validation on the data output from prepare instances
            first of two calls to normalize that I need to figure out how to remove at some point."""
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, self.remove_corr)
        self.descriptor_names = train_column_names
        train_features_pandas = pd.DataFrame(train_features)
        normalized_features_pandas = normalize(train_features_pandas, None)
        normalized_features_np = np.array(normalized_features_pandas)
        self.kxfoldvalidation(normalized_features_np, train_labels)
        return self

    def kerasmodel(self, train_data_np, test_data_np, index):
        """this is the method called for continuous models, taking in the train and test data as well as the index of
        the fold in k-fold cross validation (k)"""
        model = models.Sequential()
        #chunk 1
        model.add(layers.Dense(120, kernel_regularizer=regularizers.l1(0.001),
                               input_shape=(train_data_np.shape[1],)))
        model.add(Activation("relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        #chunk 2
        model.add(layers.Dense(60))
        model.add(Activation("relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        #chunk 3

        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        model.fit(train_data_np, test_data_np, epochs=self.epochs, batch_size=self.batch, verbose=0)
        model.save(self.temp_path + "/model" + str(index))

        stringlist = []
        model.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)


        tf.keras.backend.clear_session()
        return model

    def categorical_model(self, train_data_np, test_data_np, index):
        """ the settings for the categorical model, turns out copying the network structure of the continuous model
        works fine. Different loss, different metrics for evaluation when training than continuous, but very similar."""
        model = models.Sequential()
        #chunk 1
        model.add(layers.Dense(120, kernel_regularizer=regularizers.l1(0.001),
                               input_shape=(train_data_np.shape[1],)))
        model.add(Activation("relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        #chunk 2
        model.add(layers.Dense(60))
        model.add(Activation("relu"))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5))
        #chunk 3

        model.add(layers.Dense(1))

        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data_np, test_data_np, epochs=self.epochs, batch_size=self.batch, verbose=0)
        model.save(self.temp_path + "/model" + str(index))
        tf.keras.backend.clear_session() # I'm not sure this does anything
        return model



    def kxfoldvalidation(self, features_np, targets_np):
        """" k-fold cross validation method that creates multiple models depending on the number of folds
        parameters are the features and target numpy arrays, adds model files to the Model class as a list depending on
        the number of folds k. does the validation/train splitting based on the size of the dataset
        is endpoint specific (binary/continuous)"""

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

    def do_predictions(self, df_prediction):
        """does the predictions on the prediction dataframe taken in as a parameter. cycles through all the models,
        generates predictions, and averages those as final predictions after predictions are generated, does some
        console output of performance based on binary status."""

        train_ids, train_labels, train_features, train_column_names, isbinary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, self.remove_corr)
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, train_column_names)
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
            print('Rsq for Test Data = ', r2)
        else:
            avg_predictions_binary = np.where(avg_predictions > 0.5, 1, 0)
            BA = balanced_accuracy_score(pred_labels, avg_predictions_binary)
            print('Balanced Accuracy for Test Data =', BA)
            shutil.rmtree(self.temp_path)
            return avg_predictions_binary

        # remove the folder files
        shutil.rmtree(self.temp_path)

        return avg_predictions

    def create_models_from_binary(self, model_dropoff_path):
        for i in range(self.k):
            Modelfiles_i = self.model_files_objects[i]
            Modelfiles_i.create_model_directory_from_binary(model_dropoff_path, i)

class ModelDescription:
    """class for the model description"""
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
        self.batches = Model.batch

    def to_json(self):
        """converts the model description to json format to be stored in the database"""
        return json.dumps(self.__dict__)

def main():
    # tracks how long the model takes to train and run on tsv files wherever those are in the path.
    startTime = time.time()

    # df_training = pd.read_csv("E:\OPERA benchmark sets\LogHalfLife OPERA\LogHalfLife OPERA T.E.S.T. 5.1 training.tsv", delimiter='\t')
    # df_prediction = pd.read_csv("E:\OPERA benchmark sets\LogHalfLife OPERA\LogHalfLife OPERA T.E.S.T. 5.1 prediction.tsv", delimiter='\t')
    df_training = pd.read_csv("E:\DataSetsBenchmarkTEST_Toxicity\Mutagenicity\Mutagenicity_prediction_set-2d.csv", quotechar='"')
    df_prediction = pd.read_csv("E:\DataSetsBenchmarkTEST_Toxicity\Mutagenicity\Mutagenicity_training_set-2d.csv", quotechar='"')

    remove_log_p_descriptors = False

    model = Model(df_training, remove_log_p_descriptors)
    model.build_model()
    model.do_predictions(df_prediction)
    print(ModelDescription(model).to_json())
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))


if __name__ == "__main__":
    main()
