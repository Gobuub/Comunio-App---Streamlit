# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:03:43 2022

@author: botic
"""

import streamlit as st
import pandas as pd
from src.modelos import ComunioLib
import base64
from PIL import Image
import urllib.request

st.set_page_config(layout='centered', page_icon='⚽', page_title='Comunio Assistant')

st.title('COMUNIO ASSISTANT')
journey = 25
path = 'src/data/train/'
season = 'Season_22_23'
st.write(f'''
        # YOUR COMUNIO´S BEST FRIEND
         
        ## APP TO PREDICT PLAYER'S POINTS ON THE NEXT MATCH 
         
        If you want to predict your personal squad you can download the data of all players and upload a csv file with
        your data squad, you must use the same format of the csv file, download the full data of the {journey}rd Journey 
        below, and an example of squad.
''')


df = pd.read_csv(f"src/data/pred/comunio_J{journey}.csv")
df_all = ComunioLib.create_data_train(journey, path, season)
example = pd.read_csv("src/data/pred/example_squad.csv")


def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="j_3_data.csv">Download J {journey} Data CSV File</a>'
    return href


def squaddownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="example_squad.csv">Download Example Squad CSV File</a>'
    return href


st.set_option('deprecation.showPyplotGlobalUse', False)
st.markdown(filedownload(df_all), unsafe_allow_html=True)
st.markdown(squaddownload(example), unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():

        teams = st.sidebar.selectbox('team', df_all.Team.unique())

        players = st.sidebar.selectbox('Player', df_all.loc[df_all.Team == teams]['Player'])

        features = pd.DataFrame(df_all.loc[(df_all.Team == teams) & (df_all.Player == players)])

        return features, teams, players


    input_df, teams, players = user_input_features()

df = input_df

# Displays the user input features
st.subheader('User Input features')


if uploaded_file is not None:
    st.write(df[['Team', 'Player', 'Position', 'Points_Average', 'Value']])
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')


# Calls function to make predictions
prediction = ComunioLib.predict_rnn2(df, journey)

# Create an empty df to show the results to the customer
pred = pd.DataFrame()
pred['Position'] = df.Position
pred['Player'] = df.Player
pred['Prediction'] = prediction
pred = pred.round()
pred['Avg_last_5_Games'] = df.Avg_last_5_games

if uploaded_file is not None:

    st.sidebar.subheader('Select your Squad Lineup')
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

    img_squad = df.loc[df.Team == teams]['squad_img'].values[0]

    img_player = df.loc[df.Player == players]['img'].values[0]

    urllib.request.urlretrieve(
        img_squad,
        "image_squad.png")
    image_squad = Image.open("image_squad.png")
    urllib.request.urlretrieve(
        img_player,
        "image_player.png")
    image_player = Image.open("image_player.png")

    player_pred = pred['Prediction'].values[0]
    player_avg = df['Points_Average'].values[0]

    st.sidebar.info(df.Team.values[0])
    st.sidebar.image(image_squad)
    st.subheader('Prediction for next match')
    col1, col2, col3 = st.columns(3)  # Make 3 columns to show the info
    with col1:
        st.info(df['Player'].values[0])
        st.image(image_player)

    with col2:
        st.info('Position')
        st.subheader(df['Position'].values[0])

        st.info('Next Macht Prediction')
        if player_pred > player_avg:
            st.success(f'# {player_pred}')
        elif player_pred < player_avg:
            st.error(f'# {player_pred}')
        else:
            st.subheader(player_pred, anchor='center')

    with col3:
        st.info('Value')
        st.subheader(df['Value'].values[0])
        st.info('Points Average')
        st.subheader(df['Points_Average'].values[0])


