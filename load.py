# -*- coding: utf-8 -*-

import os
import pickle as pk
import numpy as np
import pandas as pd


## First part : load the ".csv" files in the current working directory and convert them to Panda DataFrames.--------------


os.chdir('./')
file_chdir = os.getcwd() #current folder
csv_list = []
for root, dirs, files in os.walk(file_chdir):
    for file in files:
        if os.path.splitext(file)[1] == '.csv':
            csv_list.append(file)
            
csv_list             #list of String containing the name of the different ".csv" files 

data = pd.read_csv(csv_list[0], encoding='ISO-8859-1')

#Check what the DataFrame looks like
nrow = data.shape[0] #row number
ncol = data.shape[1] #column number
data.head()          



#If you want to fuse all the ".csv" files together and mix them in a huge DataFrame : uncomment below


#df_list = []
#for csv in csv_list:
#    df = pd.read_csv(csv, encoding='ISO-8859-1')
#    df_list.append(df)
#    
#data = pd.concat(df_list)


## Second part : cleaning up the data to make it exploitable, convert into a numpy array -------------------------------------

#Delete non numeric columns
labelColumn = data[' Label']     #for any other datasets : replace ' Label' by the actual name of the LABEL column. The space char is imperative here, since it is present in the csv file.
data = data._get_numeric_data()
data['Label'] = labelColumn
data.head()


# Deleting every row containing either an Inf or a NaN value.
data.replace(np.inf, np.nan, inplace=True)
data.replace(-np.inf, np.nan, inplace=True)
data = data.dropna(axis = 0, how = 'any')
data.shape[0]

#Convert into a numpy array
data.as_matrix()


##Third part :save the DataFrame into a pickle and re-load it --------------------------

#Saving in a pickle
with open(r'./alldata.txt', 'wb') as f: #must be in binary mode!
    pk.dump(data, f)
    
data = (pd.read_pickle('./alldata.txt')).as_matrix 

