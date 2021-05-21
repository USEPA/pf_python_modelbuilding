import xgboost as xgb
import numpy as np
import pandas as pd
from models import df_utilities as DFU
from sklearn.metrics import r2_score
from sklearn.metrics import balanced_accuracy_score
import json


def trim_correlated(train_pandas, test_pandas, threshold):
    df_corr = train_pandas.corr(method='pearson', min_periods=1)
    df_not_correlated = ~(df_corr.mask(np.tril(np.ones([len(df_corr)] * 2, dtype=bool))).abs() > threshold).any()
    un_corr_idx = df_not_correlated.loc[df_not_correlated[df_not_correlated.index] == True].index
    df_out = train_pandas[un_corr_idx]
    if test_pandas is not None:
        df_test_out = test_pandas[un_corr_idx]
        return df_out, df_test_out
    else:
        return df_out, un_corr_idx

def demomain():
    df_training = pd.read_csv("LogBCF OPERA T.E.S.T. 5.1 training.tsv", delimiter='\t')
    df_prediction = pd.read_csv("LogBCF OPERA T.E.S.T. 5.1 prediction.tsv", delimiter='\t')
    train_ids, train_labels, train_features, train_column_names, isbinary = \
        DFU.prepare_instances(df_training, "training", False)

    corr_threshold = 0.95
    pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, train_column_names)
    train_features_pandas = pd.DataFrame(train_features)
    pred_features_pandas = pd.DataFrame(pred_features)
    trimmed_train_features_pandas, trimmed_test_features_pandas = trim_correlated(train_features_pandas,
                                                                                  pred_features_pandas,
                                                                                  corr_threshold)

    xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.35,
                              max_depth=5, alpha=10, n_estimators=10)
    print(trimmed_test_features_pandas)
    print(trimmed_train_features_pandas)
    xg_reg.fit(trimmed_train_features_pandas, train_labels)

    preds = xg_reg.predict(trimmed_test_features_pandas)
    r2 = r2_score(pred_labels, preds)

    print(r2)

class Model:
    def __init__(self, df_training, remove_log_p_descriptors):
        self.k = 5
        self.descriptor_names = None
        self.qsar_method = 'XGBoost'
        self.df_training = df_training
        self.remove_log_p_descriptors = remove_log_p_descriptors
        self.description = ''
        self.remove_corr = True
        self.corr_threshold = 0.95
        self.model = None
        self.is_binary = None
        self.models = None
        self.version = "1.1"
        self.learning_space = [i / 100.0 for i in range(1, 51, 5)] # LR params (0,1]
        self.colsample_bytree_space = [0.3, 0.6] # [0.3, 0.6, 0.8, 1.0]
        self.max_depth_space = [4, 7]
        self.alpha_space = [0, 0.1, 0.5, 10]
        self.n_estimators_space = [i for i in range(20, 120, 5)]
        self.modeldesc = None

    def build_model(self):
        train_ids, train_labels, train_features, train_column_names, self.is_binary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, self.remove_corr)
        self.descriptor_names = train_column_names
        self.kxfoldvalidation(train_features, train_labels)
        return self

    def get_hyperparam_models(self):
        models = dict()
        for l in self.learning_space:
            for cs in self.colsample_bytree_space:
                for alpha in self.alpha_space:
                    for md in self.max_depth_space:
                        for ne in self.n_estimators_space:
                            xg_reg = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=cs, learning_rate=l,
                                      max_depth=md, alpha=alpha, n_estimators=ne)
                            models["learnrate= ", l, "csbt =", cs, "alpha= ", alpha, "md= ", md, "ne= ", ne] = xg_reg
        return models


    def kxfoldvalidation(self, features_np, targets_np):
        k = self.k
        r2_array = []
        num_validation_samples = len(features_np) // k
        models = self.get_hyperparam_models()
        all_models_dict = dict()
        counter = 0

        for x in models.values():
            valr2_array = []
            k_models = list()
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
                        x.fit(partial_train_data, partial_train_targets)
                        preds = x.predict(val_data)
                        r2 = r2_score(val_targets, preds)
                        valr2_array.append(r2)
                        k_models.append(x)
                else:
                    break
                # need to expand for classification
            r2_array.append(np.average(valr2_array))
            all_models_dict[counter] = k_models
            counter += 1
        best_r2 = r2_array.index(max(r2_array))
        print(best_r2)
        self.models = list(all_models_dict.values())[best_r2]
        self.modeldesc = str(self.models[0]) #



    def do_predictions(self, df_prediction):
        train_ids, train_labels, train_features, train_column_names, isbinary = \
            DFU.prepare_instances(self.df_training, "training", self.remove_log_p_descriptors, self.remove_corr)
        pred_ids, pred_labels, pred_features = DFU.prepare_prediction_instances(df_prediction, train_column_names)
        predictionslist = []
        for i in range(self.k):
            model_i = self.models[i]
            predictions = model_i.predict(pred_features)
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
        return avg_predictions

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
        self.modelparamstring = Model.modeldesc

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