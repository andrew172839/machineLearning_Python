import numpy as np 
import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

df_train = pd.read_json(open('./train.json'))
print df_train.shape

df_train['num_photos'] = df_train['photos'].apply(len)
df_train['num_features'] = df_train['features'].apply(len)
df_train['num_description_words'] = df_train['description'].apply(lambda x: len(x.split(' ')))
df_train['created'] = pd.to_datetime(df_train['created'])
df_train['created_year'] = df_train['created'].dt.year
df_train['created_month'] = df_train['created'].dt.month
df_train['created_day'] = df_train['created'].dt.day

df_test = pd.read_json(open('./test.json'))
print df_test.shape

df_test['num_photos'] = df_test['photos'].apply(len)
df_test['num_features'] = df_test['features'].apply(len)
df_test['num_description_words'] = df_test['description'].apply(lambda x: len(x.split(' ')))
df_test['created'] = pd.to_datetime(df_test['created'])
df_test['created_year'] = df_test['created'].dt.year
df_test['created_month'] = df_test['created'].dt.month
df_test['created_day'] = df_test['created'].dt.day

features = ['bathrooms', 'bedrooms', 'longitude', 'latitude', 'price', 'num_photos', 'num_features', 'num_description_words', 'created_year', 'created_month', 'created_day']
X = df_train[features]
y = df_train['interest_level']

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33)

clf = LogisticRegression(C = 0.1, penalty = 'l1')
clf.fit(X_train, y_train) 

y_valid_pred = clf.predict_proba(X_valid)
print log_loss(y_valid, y_valid_pred)

X_test = df_test[features]
y_test = clf.predict_proba(X_test)

submit = pd.DataFrame()
submit['listing_id'] = df_test['listing_id']
labelToIndex = {label: i for i, label in enumerate(clf.classes_)}
for label in ['high', 'medium', 'low']:
    submit[label] = y_test[:, labelToIndex[label]]
submit.to_csv('temp.csv', index = False)
