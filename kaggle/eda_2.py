import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from subprocess import check_output
print check_output(['ls', './'])

train = pd.read_json('./train.json')
test = pd.read_json('./test.json')
print train.shape
print test.shape
print train.info()
print train.get_dtype_counts()
print train.describe()

patches, texts, autotexts = plt.pie(train.interest_level.value_counts().values, labels = ['low', 'medium', 'high'], colors = ['lightcoral', 'gold', 'lightblue'], explode = [0.1, 0, 0], autopct = '%1.1f% %', startangle = 90)
plt.title('interest level')
plt.show()

plt.scatter(range(train.shape[0]), train['price'].values)
plt.title('distribution of price')
plt.ylabel('price')
plt.show()

ulimit = np.percentile(train.price.values, 99)
train['price'].ix[train['price'] > ulimit] = ulimit
plt.scatter(range(train.shape[0]), train['price'].values)
plt.title('distribution of price')
plt.ylabel('price')
plt.show()

sns.stripplot(train['interest_level'], train['price'], jitter = True, order = ['low', 'medium', 'high'])
plt.show()

train['bedrooms'].value_counts().plot(kind = 'bar')
plt.xlabel('number of bedrooms')
plt.ylabel('number of occurences')
plt.show()

sns.stripplot(x = 'interest_level', y = 'bedrooms', data = train, jitter = True, order = ['low', 'medium', 'high'])
plt.show()

sns.stripplot(x = 'bedrooms', y = 'price', data = train, jitter = True)
plt.show()

sns.stripplot(x = 'bathrooms', y ='price' ,data = train, jitter = True)
plt.show()

sns.stripplot(x = 'interest_level', y = 'bathrooms', data = train, jitter = True, order = ['low', 'medium', 'high'])
plt.show()

sns.stripplot(train['interest_level'], train['listing_id'], jitter = True, order = ['low', 'medium', 'high'])
plt.show()
