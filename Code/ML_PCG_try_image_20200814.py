"""
# A deep learning model for predicting mechanical properties of polycrystalline graphene
# Authors : Md Imrul Reza Shishir, Mohan Surya Raja Elapolu, Alireza Tabarraei
# Affliation : Multiscale Material Modelding Labratory, Department of Mechanical Engineering and Engineering Science, The University of North Carolina at Charlotte, Charlotte, NC 28223, USA
# Corresponding Author : atabarra@uncc.edu , mshishir@uncc.edu
# Submitted to Computational Material Science, 2022
# For academic purposes only

Created on Fri Jul 24 02:27:35 2020

@author: imrul

"""

import cv2 as cv
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import KFold, train_test_split
from tensorflow.keras.optimizers import Adam, Adamax, Adadelta, Adagrad, RMSprop
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from tensorflow.keras import regularizers
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras import backend as K
import os

from Model import *

##########################################################################

initial_directory =os.getcwd()

dataset_dir ='/d-CNN_PCG/Data/'

list_file = ['New_300L_GN_2.0_6.0_25_225_trail_1','New_300L_GN_2.0_6.0_25_225_trail_2','New_300L_GN_2.0_6.0_25_225_trail_3','New_300L_GN_2.0_6.0_25_225_trail_4','New_300L_GN_2.0_6.0_25_225_trail_5','New_300L_GN_2.0_6.0_25_225_trail_6','New_300L_GN_2.0_6.0_25_225_trail_7','New_300L_GN_2.0_6.0_25_225_trail_8','New_300L_GN_2.0_6.0_25_225_trail_9','New_300L_GN_2.0_6.0_25_225_trail_10']

csv_path = []


for i in list_file:
  dataset_file = str(i)+'.csv'
  dataset_path = str(dataset_dir) + str(dataset_file)
  csv_path.append(dataset_path)
  
######################################################################################

df = pd.concat((pd.read_csv(file).assign(filename = file)
                for file in csv_path), ignore_index = True)

#########################################################################

train_images = df['image_path'].apply(lambda x: np.asarray(cv.imread(dataset_dir+ 'Image/' + x, 0)))
train_scores = np.asarray(df['grain_size'])

################################################################################

n_epoch = 500
preferred_height = 224
preferred_width = 224
n_d=1

#############################################################################
    
mm = []
for i in train_images:
    uimage = cv.resize(i, (preferred_height, preferred_width))
    mm.append(uimage)

train_images = np.asarray(mm)

#############################################################################

def shuffler(x, y):
    r_list = np.random.permutation(np.arange(x.shape[0]))
    return x[r_list], y[r_list]

def my_RMSE(y_true, y_pred):
    # y_true = K.constant(y_true) 
    # y_pred = K.constant(y_pred) 
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

############################################################################

define_rand = 12
train_images = train_images / 255.0
train_images = train_images.reshape(train_images.shape[0], preferred_height, preferred_width, n_d)
train_images, test_images, train_scores, test_scores = train_test_split(train_images, train_scores ,
                                                    test_size = 0.2,
                                                    random_state = define_rand)

#############################################################################

num = 100
params_list = np.concatenate((np.random.randint(1, 300, size=(num, 1))*0.00001,np.random.randint(1, 11, size=(num, 1))*5,),axis=1).tolist()

###############################################################################

new_filenum = 0

# Training_result_list = pd.DataFrame(columns=['Testing_rmse','Testing_r2_score','Testing_mae','Traning_rmse','Traning_r2_score','Traning_mae','Learning_Rate','Epoch','Batch_size'])
Training_result_list = []
for params_l in params_list:
    filename="BS_"+str(int(params_l[1]))+"_LR_"+str(round(params_l[0],7))
    if os.path.exists(os.path.join(initial_directory,filename)):
        continue
    else:
        os.mkdir(filename)

    Directory = os.path.join(initial_directory,filename)
    os.chdir(Directory)
    f= open(filename+'.txt', 'w+')
    model = model_cnn1(preferred_height = preferred_height, preferred_width = preferred_width) 
    model.summary()
    model.compile(loss=tf.keras.losses.mean_squared_error,
                  optimizer=Adam(learning_rate=round(params_l[0],15)),
                  metrics=[tf.keras.metrics.RootMeanSquaredError(),'mae'])
    
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(filename+'.h5', save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=5, verbose=1, epsilon=1e-4, mode='min')
    history = model.fit(train_images, train_scores, validation_split=0.25, batch_size=int(params_l[1]), epochs=n_epoch, callbacks=[earlyStopping, mcp_save,reduce_lr_loss])
    
    model.load_weights(filepath = filename+'.h5')

    result = model.evaluate(test_images, test_scores, verbose=0)

    f.write("Learning rate: %f \nEpoch: %f \nBatch Size: %f \n\n\n" % (round(params_l[0],6),(len(history.history['val_loss'])-10),int(params_l[1])))
    f.write("Test Result %r" % (result))
        
    ###############################################################
    Train_pred_scores = model.predict(train_images[0:],batch_size=1)
    df1 = pd.DataFrame({'Actual': train_scores.flatten(), 'Predicted': Train_pred_scores.flatten()})

    Test_pred_scores = model.predict(test_images[0:],batch_size=1)
    df2 = pd.DataFrame({'Actual': test_scores.flatten(), 'Predicted': Test_pred_scores.flatten()})

    ###############################################################
    Traning_rmse = np.sqrt(mean_squared_error(df1['Actual'], df1['Predicted']))
    Traning_r2_score = r2_score(df1['Actual'], df1['Predicted'])
    Traning_mae = mean_absolute_error(df1['Actual'], df1['Predicted'])

    Testing_rmse = np.sqrt(mean_squared_error(df2['Actual'], df2['Predicted']))
    Testing_r2_score = r2_score(df2['Actual'], df2['Predicted'])
    Testing_mae = mean_absolute_error(df2['Actual'], df2['Predicted'])
    
    Training_result_list.append([Testing_rmse, Testing_r2_score, Testing_mae,Traning_rmse, Traning_r2_score, Traning_mae,round(params_l[0],6),(len(history.history['val_loss'])-10),int(params_l[1])])
    
    f.write("\n\n\n Traning RMSE: %f \n Traning r2_score: %f \n Traning MAE: %s]n\n" % (Traning_rmse, Traning_r2_score, Traning_mae))
    f.write("\n\n\n Testing RMSE: %f \n Testing r2_score: %f \n Testing MAE: %s]n\n" % (Testing_rmse, Testing_r2_score, Testing_mae))
    ###############################################################
    f.write("\n\n")
    
    f.write("\n\n\n Prediction Testing Dataset\n")    
    f.write(df2.to_string())
    
    f.write("\n\n\n Prediction Training Dataset\n") 
    f.write(df1.to_string())
    
    f.close()
    os.chdir(initial_directory)

    #################################################################################################    
    
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    plt.legend(['train', 'validation'], loc='upper left', fontsize=14)
    
    plt.savefig(os.path.join(initial_directory,filename,'TrainingVsValidation.png'), bbox_inches='tight',dpi=1000)
    plt.close()
    
    ###############################################################################################
    
    max_lim = round(max(df2.max()),0)+2
    min_lim = round(min(df2.min()),0)+2
    
    df2.plot.scatter(x= 'Actual', y= 'Predicted', figsize=(10,4), label='Test Data')
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 
    plt.axis([min_lim, max_lim, min_lim, max_lim])
    plt.plot([min_lim, max_lim],[min_lim, max_lim],label='Fit')
    plt.xlabel("Actual", fontsize=10)
    plt.ylabel("Predicted", fontsize=10)
    plt.legend(fontsize=10)

    # plt.text(0.9,0.2,'RMSE: (%s)'%(Testing_rmse),fontsize=10)
    # plt.text(0.9,0.1,'R2-Score: (%s)'%(Testing_r2_score),fontsize=10)
    
    plt.savefig(os.path.join(initial_directory,filename,'Testing_performance.png'), bbox_inches='tight',dpi=1000)
    plt.close()
    
    #################################################################################################
    
    max_lim = round(max(df1.max()),0)+2
    min_lim = round(min(df1.min()),0)+2
    
    df1.plot.scatter(x= 'Actual', y= 'Predicted', figsize=(10,4), label='Training Data')
    plt.rc('xtick', labelsize=10) 
    plt.rc('ytick', labelsize=10) 
    plt.axis([min_lim, max_lim, min_lim, max_lim])
    plt.plot([min_lim, max_lim],[min_lim, max_lim],label='Fit')
    plt.xlabel("Actual", fontsize=10)
    plt.ylabel("Predicted", fontsize=10)
    plt.legend(fontsize=10)
    
    # plt.text(0.9,0.2,'RMSE: (%s)'%(Traning_rmse),fontsize=10)
    # plt.text(0.9,0.1,'R2-Score: (%s)'%(Traning_r2_score),fontsize=10)
    
    plt.savefig(os.path.join(initial_directory,filename,'Training_performance.png'), bbox_inches='tight',dpi=1000)
    plt.close()
    
    #################################################################################################    
    
    os.chdir(initial_directory)
    
    ###############################################################################################
                    
df_total = pd.DataFrame(Training_result_list,columns=['Testing_rmse','Testing_r2_score','Testing_mae','Traning_rmse','Traning_r2_score','Traning_mae','Learning_Rate','Epoch','Batch_size'])

best_Testing_score = df_total.loc[df_total['Testing_rmse'].idxmin()]
best_Traning_score = df_total.loc[df_total['Traning_rmse'].idxmin()]

CSV_filename = 'Total_file.csv' 

Txt_filename = 'Total_file.txt' 

Best_Training_Resul = 'Best_Training_Result.txt' 
Best_Testing_Resul = 'Best_Testing_Result.txt'


if os.path.exists(os.path.join(initial_directory,CSV_filename)):
    df_total.to_csv(os.path.join(initial_directory,CSV_filename), mode='a', header=False, index=False)
    df_total.to_csv(os.path.join(initial_directory,Txt_filename), sep=' ', mode='a', header=False, index=False)
    best_Traning_score.to_csv(os.path.join(initial_directory,Best_Training_Resul), sep=' ', mode='a', header=False, index=False)
    best_Testing_score.to_csv(os.path.join(initial_directory,Best_Testing_Resul), sep=' ', mode='a', header=False, index=False)

else:
    df_total.to_csv(os.path.join(initial_directory,CSV_filename), header=True, index=False)
    df_total.to_csv(os.path.join(initial_directory,Txt_filename), sep=' ', header=True, index=False)
    best_Traning_score.to_csv(os.path.join(initial_directory,Best_Training_Resul), sep=' ', header=True, index=False)
    best_Testing_score.to_csv(os.path.join(initial_directory,Best_Testing_Resul), sep=' ', header=True, index=False)

  
