# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:47:48 2022

@author: botic
"""

import pandas as pd
from src.modelos import ComunioLib
from tensorflow import keras

# df create data train until J 35

df = pd.read_csv('src/data/train/data_train.csv')

journey = 35
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

X, y, X_train, X_test, y_train, y_test ,X_train_s, X_test_s, y_train_s, y_test_s, x_scaler, y_scaler = ComunioLib.preprocess_data(df)


model = keras.models.load_model('src/comunio_rnn2.h5')

model.fit(X_train_s,
          y_train_s,
          epochs=20,
          validation_split=0.2
          )

model.save(f'src/comunio_rnn2_J{journey}.h5')
print('Model trained and saved')