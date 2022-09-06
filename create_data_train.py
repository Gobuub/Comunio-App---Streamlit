import src.modelos as cm
import pandas as pd
from datetime import datetime

path = 'src/data/train/'
season = 'Season_22_23'

old = pd.read_csv(path + 'data_train.csv')
print('old shape', old.shape)
# create new df for journey 3 date: 2022-09-06
journey = int(input(('Intoduce el número de jornada a crear: ')))
new = cm.ComunioLib.create_data_train(journey, path, season)
print('new shape', new.shape)
date = datetime.now()
print(date)

n_j = journey + 1

next_j = cm.ComunioLib.create_data_train(n_j, path, season)
next_j = next_j[['Player', 'J_Actual']]
next_j_target = next_j.rename(columns={'Player': 'Jugador', 'J_Actual': 'Target'})
df_2 = new.merge(next_j_target, how='left', left_on='Player', right_on='Jugador')
print('df_2 shape', df_2.shape)
df_2 = df_2.dropna()
print('df_2 shape Despues de dropna ', df_2.shape)

train = pd.concat([old, df_2], axis=0).dropna()
print('train shape ',train.shape)
print(len(old), print(len(df_2)), len(train))
old.to_csv(path+f"data_train_old_{str(date).split(' ')[0]}.csv", index=False)
train.to_csv(path+'data_train.csv', index=False)

