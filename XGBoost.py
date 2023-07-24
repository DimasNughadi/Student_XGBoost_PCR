import numpy as np
import pandas as pd
import pickle
import pymysql
from sklearn.metrics import accuracy_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn import preprocessing
import time

start_time = time.time()
# database mysql connection
mysql_cn = pymysql.connect(host='localhost', port=3306, user='root', passwd='', db='kelulusan')
df_raw = pd.read_sql("SELECT * FROM alumni where NIM < 2018000000;", con=mysql_cn)
df_g18 = pd.read_sql("SELECT * FROM alumni where NIM > 2018000000;", con=mysql_cn)
mysql_cn.close()

columns=['NIM', 'NAMA', 'KELAS', 'CUTI', 'KP', 'IPS1', 'IPS2', 'IPS3', 'IPS4', 'IPS5', 'IPS6', 'CO', 'KOMPEN', 'TAK' ,'STATUS']

df_raw = df_raw.reindex(columns= columns)
df_g18 = df_g18.reindex(columns= columns)

def preprocesss_kelulusan_df(df_raw):
    df_raw.loc[df_raw['IPS1'] <= 2.76, 'KATIPS1'] = '<2.76'
    df_raw.loc[(df_raw['IPS1'] > 2.76) & (df_raw['IPS1'] <= 3), 'KATIPS1'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS1'] > 3) & (df_raw['IPS1'] <= 3.5), 'KATIPS1'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS1'] > 3.5, 'KATIPS1'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS2'] <= 2.76, 'KATIPS2'] = '<2.76'
    df_raw.loc[(df_raw['IPS2'] > 2.76) & (df_raw['IPS2'] <= 3), 'KATIPS2'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS2'] > 3) & (df_raw['IPS2'] <= 3.5), 'KATIPS2'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS2'] > 3.5, 'KATIPS2'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS3'] <= 2.76, 'KATIPS3'] = '<2.76'
    df_raw.loc[(df_raw['IPS3'] > 2.76) & (df_raw['IPS3'] <= 3), 'KATIPS3'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS3'] > 3) & (df_raw['IPS3'] <= 3.5), 'KATIPS3'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS3'] > 3.5, 'KATIPS3'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS4'] <= 2.76, 'KATIPS4'] = '<2.76'
    df_raw.loc[(df_raw['IPS4'] > 2.76) & (df_raw['IPS4'] <= 3), 'KATIPS4'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS4'] > 3) & (df_raw['IPS4'] <= 3.5), 'KATIPS4'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS4'] > 3.5, 'KATIPS4'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS5'] <= 2.76, 'KATIPS5'] = '<2.76'
    df_raw.loc[(df_raw['IPS5'] > 2.76) & (df_raw['IPS5'] <= 3), 'KATIPS5'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS5'] > 3) & (df_raw['IPS5'] <= 3.5), 'KATIPS5'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS5'] > 3.5, 'KATIPS5'] = '3.50 - 4.00'

    df_raw.loc[df_raw['IPS6'] <= 2.76, 'KATIPS6'] = '<2.76'
    df_raw.loc[(df_raw['IPS6'] > 2.76) & (df_raw['IPS6'] <= 3), 'KATIPS6'] = '2.76 - 3.00'
    df_raw.loc[(df_raw['IPS6'] > 3) & (df_raw['IPS6'] <= 3.5), 'KATIPS6'] = '3.00 - 3.50'
    df_raw.loc[df_raw['IPS6'] > 3.5, 'KATIPS6'] = '3.50 - 4.00'

    df_raw.loc[df_raw['KOMPEN'] <= 25, 'KATKOMPEN'] = '<25'
    df_raw.loc[(df_raw['KOMPEN'] > 25) & (df_raw['KOMPEN'] <= 50), 'KATKOMPEN'] = '25 - 50'
    df_raw.loc[(df_raw['KOMPEN'] > 50) & (df_raw['KOMPEN'] <= 75), 'KATKOMPEN'] = '50 - 75'
    df_raw.loc[df_raw['KOMPEN'] > 75, 'KATKOMPEN'] = '>75'

    df_raw.loc[df_raw['TAK'] <= 25, 'KATTAK'] = '<25'
    df_raw.loc[(df_raw['TAK'] > 25) & (df_raw['TAK'] <= 50), 'KATTAK'] = '25 - 50'
    df_raw.loc[(df_raw['TAK'] > 50) & (df_raw['TAK'] <= 75), 'KATTAK'] = '50 - 75'
    df_raw.loc[df_raw['TAK'] > 75, 'KATTAK'] = '>75'

    return df_raw

def preprocess_kelulusan_df(df_raw):
    processed_df = df_raw.copy()
    le = preprocessing.LabelEncoder()
    processed_df.CUTI = le.fit_transform(processed_df.CUTI)
    processed_df.KP = le.fit_transform(processed_df.KP)
    processed_df.KATIPS1 = le.fit_transform(processed_df.KATIPS1)
    processed_df.KATIPS2 = le.fit_transform(processed_df.KATIPS2)
    processed_df.KATIPS3 = le.fit_transform(processed_df.KATIPS3)
    processed_df.KATIPS4 = le.fit_transform(processed_df.KATIPS4)
    processed_df.KATIPS5 = le.fit_transform(processed_df.KATIPS5)
    processed_df.KATIPS6 = le.fit_transform(processed_df.KATIPS6)
    processed_df.KATKOMPEN = le.fit_transform(processed_df.KATKOMPEN)
    processed_df.KATTAK = le.fit_transform(processed_df.KATTAK)
    processed_df.STATUS = le.fit_transform(processed_df.STATUS)
    processed_df = processed_df.drop(["NIM", "NAMA", "KELAS"], axis=1)

    return processed_df

processed_df1 = preprocesss_kelulusan_df(df_raw)
processed_df2 = preprocess_kelulusan_df(processed_df1)

processed_df1_18 = preprocesss_kelulusan_df(df_g18)
processed_df2_18 = preprocess_kelulusan_df(processed_df1_18)

columns = ['CUTI', 'KP', 'KATIPS1', 'KATIPS2', 'KATIPS3', 'KATIPS4', 'KATIPS5', 'KATIPS6', 'CO', 'KATKOMPEN', 'KATTAK', 'STATUS']

processed_df2 = processed_df2.reindex(columns=columns)
processed_df2_18 = processed_df2_18.reindex(columns=columns)

X_train = processed_df2.drop(["STATUS"], axis=1).values
X_test = processed_df2_18.drop(["STATUS"], axis=1).values
y_train = processed_df2["STATUS"].values
y_test = processed_df2_18["STATUS"].values

xgb_cl = XGBClassifier(base_score=0.5, booster='gbtree', callbacks=None,
                       colsample_bylevel=1, colsample_bynode=1, colsample_bytree=0.7,
                       early_stopping_rounds=None, enable_categorical=False,
                       eval_metric=None, feature_types=None, gamma=0.3, gpu_id=-1,
                       grow_policy='depthwise', importance_type=None,
                       interaction_constraints='', learning_rate=0.05, max_bin=256,
                       max_cat_threshold=64, max_cat_to_onehot=4, max_delta_step=0,
                       max_depth=10, max_leaves=0, min_child_weight=1,
                       monotone_constraints='()', n_estimators=100, n_jobs=0,
                       num_parallel_tree=1, predictor='auto', random_state=42)
xgb_cl.fit(X_train, y_train)
y_pred = xgb_cl.predict(X_test)
end_time = time.time()

predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print("Precision: %.2f%%" % (precision * 100.0))
print("Recall: %.2f%%" % (recall * 100.0))

# Save the time output to a file using pickle
time_output = {"fit_time": end_time - start_time}
with open("time_output.pkl", "wb") as f:
    pickle.dump(time_output, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(xgb_cl, f)

with open('processed_df2.pkl', 'wb') as f:
    pickle.dump(processed_df2, f)