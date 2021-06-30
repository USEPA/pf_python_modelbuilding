import time
import json

from xgboost import XGBRegressor, XGBClassifier
from models import df_utilities as dfu


class Model:
    """Trains and makes predictions with an out-of-the-box XGBoost model"""
    def __init__(self, df_training, remove_log_p_descriptors):
        """Initializes the XGB model with provided data in pandas dataframe"""
        self.model = None
        self.descriptor_names = None
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.is_binary = None  # Set automatically when training data is loaded
        self.df_training = df_training
        self.version = '1.0'
        self.qsar_method = 'XGBoost'
        self.description = 'python implementation of extreme gradient boosting ' \
                           '(https://xgboost.readthedocs.io/en/latest/get_started.html)'

    def build_model(self):
        """Trains the XGB model on provided data"""
        t1 = time.time()

        # Call prepare_instances without removing correlated descriptors
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            dfu.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, False)

        # Use columns selected by prepare_instances (in case logp descriptors were removed)
        self.descriptor_names = train_column_names

        if self.is_binary:
            self.model = XGBClassifier(disable_default_eval_metric=True, eval_metric='auc')
        else:
            self.model = XGBRegressor()
        # Train the model on training data
        self.model.fit(train_features, train_labels)

        print('Score for Training data = ', self.model.score(train_features, train_labels))

        # Save space in database
        self.df_training = None

        t2 = time.time()
        print('Time to train model  = ', t2 - t1, 'seconds')

        # Return built model
        return self

    def do_predictions(self, df_prediction):
        """Makes predictions using the trained model"""
        # Prepare prediction instances using columns from training data
        pred_ids, pred_labels, pred_features = dfu.prepare_prediction_instances(df_prediction, self.descriptor_names)

        # Makes predictions
        predictions = self.model.predict(pred_features)

        print('Score for Test data = ', self.model.score(pred_features, pred_labels))

        # Return predictions
        return predictions


class ModelDescription:
    def __init__(self, model):
        """Describes parameters of the specific model as built"""
        self.is_binary = model.is_binary
        self.remove_log_p_descriptors = model.remove_log_p_descriptors
        self.version = model.version
        self.qsar_method = model.qsar_method
        self.description = model.description

    def to_json(self):
        """Returns description as a JSON"""
        return json.dumps(self.__dict__)


def main():
    """
    Code to run from text files rather than webservice
    :return:
    """
    endpoint = 'Octanol water partition coefficient'
    # endpoint = 'Water solubility'
    # endpoint = 'Melting point'

    descriptor_software = 'T.E.S.T. 5.1'
    # descriptor_software = 'Padelpy webservice single'

    folder = 'C:/Users/GSincl01/OneDrive - Environmental Protection Agency (EPA)/Python/pf-python-modelbuilding/data/'
    folder += 'DataSetsBenchmark/' + endpoint + ' OPERA/'
    # folder += 'DataSetsBenchmarkTEST_Toxicity/' + endpoint + '/'
    training_file_name = endpoint + ' OPERA ' + descriptor_software + ' training.tsv'
    prediction_file_name = endpoint + ' OPERA ' + descriptor_software + ' prediction.tsv'
    # training_file_name = endpoint + '_training_set-2d.csv'
    # prediction_file_name = endpoint + '_prediction_set-2d.csv'
    training_tsv_path = folder + training_file_name
    prediction_tsv_path = folder + prediction_file_name

    # Parameters needed to build model:
    remove_log_p_descriptors = False

    df_training = dfu.load_df_from_file(training_tsv_path, sep='\t')
    df_prediction = dfu.load_df_from_file(prediction_tsv_path, sep='\t')

    # df_training = pd.read_csv(training_tsv_path, sep='\t')
    # df_prediction = pd.read_csv(prediction_tsv_path, sep='\t')

    model = Model(df_training, remove_log_p_descriptors)
    model.build_model()

    print(ModelDescription(model).to_json())
    model.do_predictions(df_prediction)


if __name__ == "__main__":
    main()
