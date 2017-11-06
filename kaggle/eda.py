import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

train_data = pd.read_json('./train.json')
train_data.isnull().sum()

##########################################################################################
sns.countplot(train_data.interest_level, order = ['low', 'medium', 'high'])
plt.xlabel('interest level')
plt.ylabel('number of occurrences')
plt.show()

sns.countplot(train_data.bathrooms)
plt.xlabel('number of bathrooms')
plt.ylabel('number of occurrences')
plt.show()

##########################################################################################
sns.barplot(x = 'interest_level', y = 'bathrooms', data = train_data, order = ['low', 'medium', 'high'])
plt.xlabel('interest level')
plt.ylabel('average number of bathrooms')
plt.show()

print train_data.display_address.value_counts().nlargest(10)

print train_data.latitude.nlargest(10)
print train_data.latitude.nsmallest(10)
print train_data.longitude.nlargest(10)
print train_data.longitude.nsmallest(10)

sns.lmplot(x = 'longitude', y = 'latitude', fit_reg = False, hue = 'interest_level', hue_order = ['low', 'medium', 'high'],
data = train_data[(train_data.longitude > train_data.longitude.quantile(0.005))
&(train_data.longitude < train_data.longitude.quantile(0.995))
&(train_data.latitude > train_data.latitude.quantile(0.005))                           
&(train_data.latitude < train_data.latitude.quantile(0.995))])
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.show()
