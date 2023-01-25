# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:47:48 2022

@author: botic
"""

import pandas as pd
from src.modelos import ComunioLib
import pickle
from tensorflow import keras
from sklearn.metrics import mean_squared_error as mse

# df create data train until J 3

df = pd.read_csv('src/data/train/data_train.csv')
# last update 25/01/2023
journey = 17
# TO ADD NEW JOURNEY

'''

Example for next journey:
    
journey = 35

df_2 = comunio_pred_lib.create_data_train(journey)

n_j = journey+1
print('Siguiente jornada',n_j)
next_j = comunio_pred_lib.create_data_train(n_j)
next_j = next_j[['Player','J_Actual']]
next_j_target= next_j.rename(columns={'Player': 'Jugador', 'J_Actual':'Target'})
df_3 = df_2.merge(next_j_target, how='left', left_on='Player', right_on='Jugador')
df_3 = df_3.dropna()
df = df.append(df_3)

df.to_csv('src/data/train/data_train.csv', index=False)

'''

# CREATE THE VARIABLES TO TRAIN OUR MODEL

X, y, X_train, X_test, y_train, y_test, X_train_s, X_test_s,\
    y_train_s, y_test_s, x_scaler, y_scaler = ComunioLib.preprocess_data(df)


model = keras.models.load_model(f'src/comunio_rnn2_J{journey-1}_temp_22-23.h5')

model.fit(X_train_s,
          y_train_s,
          batch_size=16,
          epochs=20,
          validation_split=0.2
          )

model.save(f'src/comunio_rnn2_J{journey}_temp_22-23.h5')
print('Model trained and saved')
pickle.dump(x_scaler, open('src/x_scaler.model', 'wb'))
print('x_scaler saved')
pickle.dump(y_scaler, open('src/y_scaler.model', 'wb'))
print('y_scaled saved')
