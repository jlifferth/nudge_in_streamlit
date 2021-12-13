# this script presents user predictions in a simple streamlit environment
# obtains data from user0 "0_master_df.scv"

import streamlit as st
import pandas as pd
import altair as alt
# from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import plotly.express as px

dataframe_path = 'data/glucose.csv'


def clean_df(df_in):
    # combine date and time columns, set datetime as index
    # df_in = df_in.set_index('datetime')
    df_in = pd.read_csv(df_in)
    df_in = df_in.drop(columns=['Unnamed: 0'])
    df_in['datetime'] = pd.to_datetime(df_in['datetime'])
    # print(df_in.head())
    return df_in


df = clean_df(dataframe_path)

col01, col02, col03 = st.columns(3)
with col01:
    st.header('Nudge')
    st.subheader('by Longevity Solutions')
    st.text('')
with col02:
    st.text('')
with col03:
    st.text('')
    logo_path = 'data/Longevity Solutions Logo.png'
    st.image(logo_path, width=100)

sim_counter = -int(len(df) - 10)
sim_counter = - 1
current_glucose = df.iat[sim_counter, 0]
last_glucose = df.iat[sim_counter - 6, 0]
last_glucose = (round(last_glucose, 1))
glucose_change = current_glucose - last_glucose
glucose_change = str(round(glucose_change, 1))

# place metric widget center
# col21, col22, col23, col24 = st.columns(4)
# with col21:
#     st.text('')
# with col22:
#     st.metric(label="Current Heart Rate", value='Unavailable',
#               delta=str(0) + ' change in past hour')
# with col23:
#     st.metric(label="Current Glucose Level", value=str(current_glucose) + ' mg/dL',
#               delta=glucose_change + ' mg/dL in past hour')
# with col24:
#     st.text('')

col22, col23 = st.columns(2)
with col22:
    st.metric(label="Current Heart Rate", value='Unavailable',
              delta=str(-5) + ' change in past hour')
with col23:
    st.metric(label="Current Glucose Level", value=str(current_glucose) + ' mg/dL',
              delta=glucose_change + ' mg/dL in past hour')

# import prediction models
from model_training_functions import train_model
prediction, y_test, summary = train_model(df_path=dataframe_path)

# display standard plot
alt_chart = alt.Chart(df, title="Your glucose this week").mark_line().encode(
    x='datetime',
    y='glucose',
).properties(
    width=700,
    height=350)

# print(prediction.shape, type(prediction))
# print(prediction[:10])
# print(y_test.shape, type(y_test))
# print(y_test[:10])

# my_array = np.array([prediction, y_test])
array_0 = prediction.ravel()
# print('array 0 : ', array_0.shape)
array_1 = y_test.ravel()
# print('array 1 : ', array_0.shape)
series_0 = pd.Series(array_0)
# print('Series 0 : ', series_0.shape)
series_1 = pd.Series(array_1)
# print('Series 1 : ', series_1.shape)

# display prediction plot
my_df = pd.concat([series_0, series_1], axis=1)
my_df.columns = ['prediction', 'actual']
plotly_chart = px.line(my_df, title='Nudge predictions this week')
plotly_chart.update_layout(xaxis_title="Time",
                           yaxis_title="glucose",)

# toggle_chart = st.button('prediction chart')
# toggle_chart = st.radio('Select chart', ['historical', 'predictions'])
st.session_state['chart'] = st.radio('Select chart', ['historical', 'predictions'])

if st.session_state['chart'] == 'historical':
    st.altair_chart(alt_chart)
else:
    st.plotly_chart(plotly_chart)
# if toggle_chart:
#     toggle_chart = st.button('glucose chart')
#     st.plotly_chart(plotly_chart)
# else:
#

# provide weekly summary
st.text(summary)

# IGNORE EVERYTHING BELOW
# useless code bit graveyard but I'm too sentimental to say goodbye

# display prediction plots

# FAILED
# dataset = pd.DataFrame({'date': pd.date_range('2019-01-01', freq='5m', periods=len(prediction)),
#                         'prediction': prediction,
#                         'actual': y_test},
#                        columns=['prediction', 'actual'])
# dataset.set_index(dataset['date'])
# pred_chart_0 = alt.Chart(dataset).transform_fold(
#     ['date', 'prediction', 'actual']
# ).mark_bar()
#
# st.altair_chart(pred_chart_0)

# df_pred = pd.DataFrame({
#     'Date': pd.date_range('2019-01-01', freq='D', periods=len(prediction)),
#     'prediction': prediction,
#     'actual': y_test,
# })

# print(df_pred)


# pred_source = prediction, y_test
# pred_source = pd.DataFrame(prediction[0], y_test[0], columns=[''])
# print('type = ', type(pred_source))
# pred_chart = alt.Chart(pred_source).mark_line(point=True).encode(
#     alt.X('datetime', title='Time', sort=None),
#     alt.Y('glucose', title='glucose'),
# )
# st.altair_chart(pred_chart)


# fig = sns.lineplot(x=df['datetime'], y=df['glucose'])
# plt.style.use('seaborn')
# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1)
# ax.plot(df["datetime"],
#         df["glucose"],)
# # ax.title('glucose this week')
# ax.set_title('glucose this week')
# ax.set_facecolor('xkcd:black')
# ax.can_zoom
# # plt.rcParams['axes.facecolor'] = 'black'
# ax.set_xlabel("Time")
# ax.set_ylabel("Glucose")
# st.pyplot(fig)

# source = pd.DataFrame(df["glucose"])
# line_chart = alt.Chart(source).mark_line(interpolate='basis').encode(
#     alt.X('glucose', title='Time'),
#     color='category:N').properties(title='Your glucose levels this week')
#
# st.altair_chart(line_chart)

# while sim_counter < -1:
#     print(sim_counter)
#     current_glucose = df.iat[sim_counter, 0]
#     last_glucose = df.iat[sim_counter - 1, 0]
#     last_glucose = str(round(last_glucose, 1))
#     glucose_change = current_glucose - last_glucose
#     glucose_change = str(round(glucose_change, 1))
#     sim_counter += 1
#     if sim_counter >= -1:
#         sim_counter = -int(len(df) - 10)
#     st.metric(label="Your Current Glucose", value=str(current_glucose) + ' mg/dL',
#                   delta=glucose_change + ' mg/dL')
#     time.sleep(5)
#
# st.markdown("<h1 style='text-align: left; color: black;'>Nudge</h1>", unsafe_allow_html=True)
# st.markdown("<h3 style='text-align: left; color: black;'>By Longevity Solutions</h1>", unsafe_allow_html=True)
# st.text('')
# st.text('')

# plt.plot(x=df['datetime'], y=df['glucose'])
# st.pyplot(plt)

# chart = (
#         alt.Chart(
#             data=df,
#             title="Your title",
#         )
#         .mark_line()
#         .encode(
#             x=alt.X("capacity 1", axis=alt.Axis(title="Capacity 1")),
#             y=alt.Y("capacity 2", axis=alt.Axis(title="Capacity 2")),
#         )
# )
#
# st.altair_chart(chart)


# col21, col22, col23 = st.columns(3)
# with col21:
#     st.text('\n')
# with col22:
#     st.text('\nYour glucose this week')
# with col23:
#     st.text('\n')
#
# df_data = pd.DataFrame(
#     df['glucose'],
#     columns=['glucose'])
# st.line_chart(df_data)

# sns.lineplot(x=df['datetime'], y=df['glucose'])
# # plt.title('your glucose levels this week')
# plt.ylabel('blood glucose (mg/dl)')
# sns.set_style('whitegrid')
# plt.xticks(rotation=60)
# plt.show()
