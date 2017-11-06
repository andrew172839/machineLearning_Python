import numpy as np 
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

df = pd.read_json(open('./train.json'))
print df.shape

df['num_photos'] = df['photos'].apply(len)
df['num_features'] = df['features'].apply(len)
df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))
df['created'] = pd.to_datetime(df['created'])
df['created_year'] = df['created'].dt.year
df['created_month'] = df['created'].dt.month
df['created_day'] = df['created'].dt.day

features = ['bathrooms', 'bedrooms', 'longitude', 'latitude', 'price', 'num_photos', 'num_features', 'num_description_words', 'created_year', 'created_month', 'created_day']
X = df[features]
y = df['interest_level']
print X.head()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33)

clf = GradientBoostingClassifier(n_estimators = 1100)
clf.fit(X_train, y_train) 
y_valid_pred = clf.predict_proba(X_valid)
print log_loss(y_valid, y_valid_pred)

df = pd.read_json(open('./test.json'))
print df.shape

df['num_photos'] = df['photos'].apply(len)
df['num_features'] = df['features'].apply(len)
df['num_description_words'] = df['description'].apply(lambda x: len(x.split(' ')))
df['created'] = pd.to_datetime(df['created'])
df['created_year'] = df['created'].dt.year
df['created_month'] = df['created'].dt.month
df['created_day'] = df['created'].dt.day
X = df[features]
y = clf.predict_proba(X)

submit = pd.DataFrame()
submit['listing_id'] = df['listing_id']
labelToIndex = {label: i for i, label in enumerate(clf.classes_)}
for label in ['high', 'medium', 'low']:
    submit[label] = y[:, labelToIndex[label]]
submit.to_csv('temp.csv', index = False)
