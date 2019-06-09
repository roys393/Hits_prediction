import numpy as np
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,BatchNormalization,Activation,MaxPooling1D
from math import sqrt
import pandas as pd
import csv 
from matplotlib import pyplot as plt
import keras.callbacks
from keras import optimizers
import mpld3
import time
mpld3.enable_notebook()


"""
This module creates one AI model which will be used to predict the number of hits in trivago page. First it will
read the datafile and clean the data by encoding and deleting nan etc. Then it will separate the labeled and unlabeled 
data for training and testing. AI model will be trained with the labeled data. Then it will be validated as well. After
that model will be used to predict the hits values of the unknown rows.

"""
def read_and_preprocess_whole_data(filepath):  
    
    """
     read and Pre-process the given dataset by removing nan, label encoding strings and calculating some score values for column 'path_id_set'
    :param filepath: Path to the given csv data file which will be preprocessed

    :returns: Pandas DataFrame of preprocessed data.
    
    """ 
    # read the original data from the filepath
    original_data = pd.read_csv(filepath,index_col = 'row_num',sep=';',header=0)
    
    for column in original_data.columns[original_data.isna().any()].tolist():        
    
        original_data[column] = original_data[column].fillna('0')
    
    # Collect all the entries of column path_id_set into one list    
    path_id_set_ToList = original_data['path_id_set'].tolist()
    
    # Now try to create one frequency score dict inorder to check which path accessed how much in whole dataset
    freq_score = {} 
    
    # freq_score will contain key = ip_path and val = num_of_occurrances in all session in dataset
    for ipvals in path_id_set_ToList:        
        for item in ipvals.split(';'):            
            if (item in freq_score):
                
                freq_score[item] += 1
            else: 
                freq_score[item] = 1
     
    # create a score list; which contains score of each row in path_id_set col entry
    # (e.g., first row of path_id_set col is '0;31624'. So it will search the score of 0 and 31624 in frequency_score dict and add it to make a final score for the row) 
    # intuition of this logic: If user is visited the top most frquent pages in all sessions, then to check how it is effecting the trivago hits
    # it could be possible when users visit most frequent pages they concentrate on something else on that page rather than trivago.          
    scorelist=[]           
    for item in original_data['path_id_set']:
              
        score = sum([freq_score[ip] for ip in item.split(';')])
        scorelist.append(score)
    
    original_data['path_id_set'] = scorelist
    
    # Change the weekdays into numbers instead of Str
    original_data['day_of_week'] = [time.strptime(day, "%A").tm_wday for day in original_data['day_of_week'].tolist()]
    
    # encode the locale strings using scikit learn's locale package
    le = LabelEncoder()
    original_data['locale'] = le.fit_transform(original_data['locale'])
    
    # sort the whole dataframe w.r.t. index
    original_data.sort_index(inplace=True) 
    
    # Except hits column check other columns in datafile and remove \N
    for column in original_data.columns.tolist()[:-1]:    
        original_data[column]= original_data[column].replace('\\N', '0')    
        
    return original_data


original_data = read_and_preprocess_whole_data(r'ML_Data_Scientist_Case_Study_Data.csv')
unlabeled_data = original_data.loc[original_data['hits'] == '\\N'] 
labeled_data = original_data.loc[original_data['hits'] != '\\N']


values = labeled_data.values
# ensure all data is float
values = values.astype('float32')

# Scale the data and fit in Standard Scaler
scaler = StandardScaler()
scaled = scaler.fit_transform(values)

# split into train and test input and outputs; 20 % of data used for test

train = scaled[:-92900,:]
test = scaled[-92900:,:]

# Split the train and test data with input and output

train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# using tensor board to see proper visualizations
tbCallBack = keras.callbacks.TensorBoard(log_dir=r'D:\Python_projects\file\dat\new_lstm_with_static\logs\without_peak', histogram_freq=0, 
                                         write_graph=True, write_images=True)

custom_optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.00001, amsgrad=False)

# Building a AI model in keras
model = Sequential()

model.add(Dense(40, input_dim=train_X.shape[1], init='normal',activation='tanh'))
model.add(Dropout(0.2))
model.add(Dense(20,activation='sigmoid'))
model.add(Dense(1,activation='linear'))
model.compile(loss = 'mse', optimizer=custom_optimizer,metrics = ['mse'])  

# fit network
history = model.fit(train_X, train_y, epochs=400, batch_size=472, 
                    validation_split=0.15, verbose=1, callbacks=[tbCallBack])
					
					
##################### Testing with labeled data #######################################					
test_prediction = model.predict(test_X)

# invert scaling for forecast

inv_test_prediction = np.concatenate(( test_X,test_prediction), axis=1)

inv_test_prediction = scaler.inverse_transform(inv_test_prediction)

inv_test_prediction = inv_test_prediction[:,-1]

 

# invert scaling for actual

inv_y = scaler.inverse_transform(test)

inv_y = inv_y[:,-1]

# calculate RMSE

rmse = sqrt(mean_squared_error(inv_y, inv_test_prediction))

print('Test RMSE: %.3f' % rmse)

					
#################################### Prediction unlabeled #####################################################

# Inorder to test first remove the \N values in hits column and replace with 0 inorder to scale it
for column in unlabeled_data.columns.tolist():
    
    unlabeled_data[column]= unlabeled_data[column].replace('\\N', '0')

# Scaling the test data using fitted scaler
unlabeled_scaled_data = scaler.transform(unlabeled_data.values)

# divide test data into input and output to the network
unlabeled_test_X, unlabeled_test_y = unlabeled_scaled_data[:, :-1], unlabeled_scaled_data[:, -1]
# prediction from model
hit_prediction = model.predict(unlabeled_test_X)
# invert scaling for forecast
inv_hit = np.concatenate((unlabeled_test_X,hit_prediction), axis=1)
inv_hit = scaler.inverse_transform(inv_hit)
inv_hit = np.rint(inv_hit[:,-1])
index_list = unlabeled_data.index.values

   
# creating two lists of named column to create an excel
row_num = index_list.tolist()
hits = index_list.tolist()

# writing the lists into an excel
with open(r'result.csv', "w", newline='') as infile:
   writer = csv.writer(infile)
   writer.writerow(["row_num", "hits"])    #Write Header
   for i in zip(row_num, hits):
       writer.writerow(i)   



    
