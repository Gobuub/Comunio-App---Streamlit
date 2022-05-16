# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:03:43 2022

@author: botic
"""

import streamlit as st
import pandas as pd
from src.modelos import ComunioLib
import base64

st.title('COMUNIO ASSISTANT')

st.write('''
         # YOUR COMUNIO´S BEST FRIEND
         
         ## APP TO PREDICT PLAYER'S POINTS ON THE NEXT MATCH 
         
         If you want to predict your personal squad you can download the data of all players and upload a csv file with your
         data squad,  you must use the same format of the csv file, with the
''')

df = pd.read_csv("src/data/pred/comunio_J35.csv")
df_all = ComunioLib.create_data_train(35)
example = pd.read_csv("src/data/pred/example_squad.csv")


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="example_squad.csv">Download Example Squad CSV File</a>'
    return href


st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_all), unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        teams = st.sidebar.selectbox('team', df_all.Team.unique())
        players = st.sidebar.selectbox('Player', df_all.loc[df_all.Team == teams]['Player'])

        features = pd.DataFrame(df_all.loc[(df_all.Team == teams) & (df_all.Player == players)])

        return features


    input_df = user_input_features()

df = input_df

# Displays the user input features
st.subheader('User Input features')
print(df[['Team', 'Player', 'Position', 'Points_Average', 'Value']].columns)

if uploaded_file is not None:
    st.write(df[['Team', 'Player', 'Position', 'Points_Average', 'Value']])
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')

    st.write(df[['Team', 'Player', 'Position', 'Points_Average', 'Value']])

# Call function to make predictions
prediction = ComunioLib.predict_rnn2(df)

# Create an empty df to show the results to the customer
pred = pd.DataFrame()
pred['Position'] = df.Position
pred['Player'] = df.Player
pred['Prediction'] = prediction
pred = pred.round()
pred['Avg_last_5_Games'] = df.Avg_last_5_games

if uploaded_file is not None:

    df = st.sidebar.selectbox('Defenders', (3, 4, 5))
    md = st.sidebar.selectbox('Mid', (4, 5, 3))
    dl = st.sidebar.selectbox('Forward', (3, 2, 1))
    st.subheader('Prediction for your squad')
    st.write(pred[['Position', 'Player', 'Prediction']])
    st.subheader('Eleven´s Initial Recommended')
    eleven = ComunioLib.once_ideal_rnn2(pred, df, md, dl)
    if df + md + dl == 10:
        st.subheader(f'Lineup {df}-{md}-{dl}')
        st.write(eleven)
    else:
        st.write('Input a right Lineup, example : 3-4-3, 4-4-2, ...')
else:
    st.subheader('Prediction for next match')
    st.write(pred)
