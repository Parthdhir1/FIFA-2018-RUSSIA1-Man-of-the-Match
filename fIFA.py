# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 18:48:09 2018

@author: Parth
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Adding the data into df
df= pd.read_csv('FIFA 2018 Statistics.csv')
df.head()
df.info()

#removing the columns which are not required
df= df.drop(['Own goals','1st Goal','Own goal Time','Date','Team','Opponent','Round'],axis=1)

#making the dummi variable PSO
ds= pd.get_dummies(df['PSO'],columns=['P.Shot(Yes)','P.Shot(No)'],drop_first= True)
de= pd.get_dummies(df['Man of the Match'],drop_first=True)
de.head()
ds.info()
df['P.Shot(Yes)']= ds
df= df.drop(['PSO'],axis=1)
df['MOM']=de
df= df.drop(['Man of the Match'],axis=1)

#splitting the data and making it into 2-d
X = df.drop(['MOM'],axis=1).values
y = df.loc[:,'MOM'].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

# Initialising the ANN
classifier = Sequential()

classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 19))
classifier.add(Dropout(p = 0.2))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 10, kernel_initializer = 'uniform', activation = 'relu'))
classifier.add(Dropout(p = 0.2))

# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10,nb_epoch = 100)
X_train.info()