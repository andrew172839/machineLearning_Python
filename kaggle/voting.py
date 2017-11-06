import numpy as np 
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

df_train = pd.read_json(open('./train.json'))
print df_train.shape

df_train['num_photos'] = df_train['photos'].apply(len)
df_train['num_features'] = df_train['features'].apply(len)
df_train['num_description_words'] = df_train['description'].apply(lambda x: len(x.split(' ')))
df_train['bedrooms_bathrooms'] = df_train['bedrooms'] / (df_train['bathrooms'] + 1)
df_train['price_bedrooms'] = df_train['price'] / (df_train['bedrooms'] + 1)
df_train['price_bathrooms'] = df_train['price'] / (df_train['bathrooms'] + 1)
df_train['created'] = pd.to_datetime(df_train['created'])
df_train['created_year'] = df_train['created'].dt.year
df_train['created_month'] = df_train['created'].dt.month
df_train['created_day'] = df_train['created'].dt.day

df_test = pd.read_json(open('./test.json'))
print df_test.shape

df_test['num_photos'] = df_test['photos'].apply(len)
df_test['num_features'] = df_test['features'].apply(len)
df_test['num_description_words'] = df_test['description'].apply(lambda x: len(x.split(' ')))
df_test['bedrooms_bathrooms'] = df_test['bedrooms'] / (df_test['bathrooms'] + 1)
df_test['price_bedrooms'] = df_test['price'] / (df_test['bedrooms'] + 1)
df_test['price_bathrooms'] = df_test['price'] / (df_test['bathrooms'] + 1)
df_test['created'] = pd.to_datetime(df_test['created'])
df_test['created_year'] = df_test['created'].dt.year
df_test['created_month'] = df_test['created'].dt.month
df_test['created_day'] = df_test['created'].dt.day

features = ['bedrooms', 'bathrooms', 'longitude', 'latitude', 'price', 'num_photos', 'num_features', 'num_description_words', 'bedrooms_bathrooms', 'price_bedrooms', 'price_bathrooms', 'created_year', 'created_month', 'created_day']
X = df_train[features]
y = df_train['interest_level']
print X.head()

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.33)

clf1 = RandomForestClassifier(n_estimators = 800)
clf2 = GradientBoostingClassifier(n_estimators = 1000)
eclf = VotingClassifier(estimators = [('rf', clf1), ('gb', clf2)], voting = 'soft', weights = [1, 1])
eclf.fit(X_train, y_train) 

y_valid_pred = eclf.predict_proba(X_valid)
print log_loss(y_valid, y_valid_pred)

X_test = df_test[features]
y_test = eclf.predict_proba(X_test)
print X_test.head()

submit = pd.DataFrame()
submit['listing_id'] = df_test['listing_id']
labelToIndex = {label: i for i, label in enumerate(eclf.classes_)}
for label in ['high', 'medium', 'low']:
    submit[label] = y_test[:, labelToIndex[label]]
submit.to_csv('temp.csv', index = False)
