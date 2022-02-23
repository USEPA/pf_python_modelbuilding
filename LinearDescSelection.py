import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn import preprocessing
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from collections import defaultdict
from models import df_utilities as DFU

"""
training DF - whatever comes straight out of pd.read_csv("opera path.tsv")
returns a list of what it considers important features
"""
def generatelinearlist(df):
    df_labels = df[df.columns[1]]
    labels = np.array(df_labels)
    ids = np.array(df[df.columns[0]])
    col_name_id = df.columns[0]
    col_name_property = df.columns[1]
    df = df.drop(col_name_property, axis=1)
    df = df.drop(col_name_id, axis=1)
    drop_column_names = []
    # Remove constant descriptors:
    df = DFU.do_remove_constant_descriptors(df, drop_column_names)
    scaler = preprocessing.StandardScaler().fit(df)
    X_scaled = scaler.transform(df)
    corr = spearmanr(X_scaled).correlation
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    cluster_ids = hierarchy.fcluster(dist_linkage, 0.5, criterion="distance")
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    named_ward_features = [list(df.columns)[i] for i in selected_features]
    ward_subset_df = df.iloc[:,selected_features]
    mlr = LinearRegression()
    mlr.fit(ward_subset_df, labels)
    # hashtable association of regression coefficients and their variable names
    coef_dict = {}
    for coef, feat in zip(mlr.coef_, named_ward_features):
        coef_dict[feat] = coef
    # sorts hashtable
    dic2 = dict(sorted(coef_dict.items(), key=lambda x: -abs(x[1])))
    res = list(dic2.keys())[0:11]
    print(res)
    return res

#

