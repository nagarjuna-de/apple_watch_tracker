from operator import index
import re
from turtle import width
import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image
import re
import data_handler as dh
import data_train as trm
import missingno as msno

### model
st.title('FIT tracker')
st.subheader('FItness and Transport mode tracker')
st.write("Team - Apple Watch")
st.write('Varun Kota', 'Omolara', 'Nagarjuna')
st.markdown('The Aim is to develop a fitness and user transport mode detection software that is able to be used in plug and play style into most apps and smart watches.')
st.markdown('One of the main ideas behind the project is to facilitate the Transport mode detection and calorie counting and make it more precise.')
st.write("we had given a raw data, with this we need to train our ML models and try to predict outcomes for the user..")

st.subheader('CLI')
### predicting
uploaded_file = st.file_uploader('choose file that you want to predict')
data = pd.read_csv(uploaded_file)
if st.button('show data in the csv file'):
    st.dataframe(data.head(300))

### renaming columns
def col_names_change(data):
    data.drop(['Unnamed: 0','id', 'activityrecognition#0'],axis=1, inplace=True)
    data_col = ['time','activityrecognition_1']
    for i in data.columns[2:58]:
        b = i.split('.')[2].split('#')
        data_col.append(f'{b[0]}_{b[1]}')

    for i in data.columns[58:66]:
        b = i.split('#')
        data_col.append(f'{b[0]}_{b[1]}')
    
    for i in data.columns[66:]:
        data_col.append(f'{i}')


    # data_col.append('target')
    # data_col.append('user')
    data = pd.DataFrame.from_records(data.values)
    data.columns = data_col

    return data

data = col_names_change(data)
data_col = ['time', 'activityrecognition_1', 'accelerometer_mean','accelerometer_std', 'game_rotation_vector_mean',
            'game_rotation_vector_std', 'gravity_mean', 'gravity_std',
            'gyroscope_mean', 'gyroscope_std', 'gyroscope_uncalibrated_mean',
            'gyroscope_uncalibrated_std', 'light_mean', 'linear_acceleration_mean',
            'linear_acceleration_std', 'magnetic_field_mean', 'magnetic_field_std',
            'magnetic_field_uncalibrated_mean', 'magnetic_field_uncalibrated_std',
            'orientation_mean', 'orientation_std', 'rotation_vector_mean',
            'rotation_vector_std', 'sound_mean', 'speed_mean',
            'accelerometer_min', 'accelerometer_max', 'gyroscope_uncalibrated_min',
            'gyroscope_uncalibrated_max', 'linear_acceleration_min',
            'linear_acceleration_max']

data = data[data_col]
ndt = np.asarray(data['time'])


oet, scaler, knn, x_train, x_test, y_train, y_test = dh.get_data(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Datasets\dataset_0.5sec.csv', 26000)

test = knn.transform(data)

test2 = scaler.transform(test)

lgbm = trm.train_model(x_train, y_train)

lgbm_pred = lgbm.predict(test2)
lgbm_pred = oet.inverse_transform(lgbm_pred.reshape(-1,1))

final_df = pd.DataFrame()
final_df['transport_mode'] = lgbm_pred
final_df['time'] = ndt
for i in final_df['transport_mode']:
    


if st.button('FIT Tracker'):
    st.dataframe(lgbm_pred)

################EDA
st.subheader('EDA & Approach')
#data = pd.read_csv(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Datasets\dataset_0.5sec.csv')

st.write('1.short intro to Raw dataset.')
df = pd.read_csv(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Datasets\dataset_0.5sec.csv')
st.set_option('deprecation.showPyplotGlobalUse', False)
fig = msno.matrix(df, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0),inline=True)

st.pyplot(fig)
st.write('There are lot of missing values in the dataset.')
###############################################
st.write("2.Does targets are Balanced?")
df1 = df['target']
fig = px.histogram(df1, x='target', nbins=100 )
fig.update_layout(title_text = 'Balance of target classes', title_x=0.5)

st.plotly_chart(fig)
st.write('The target classes in the balanced dataset are well Balanced.')

################################################33
st.write('3.Identifying the user Behaviours')

df3 = df.groupby(['user', 'target'], as_index=False).count()
fig = plt.figure(figsize=(16, 8))
sns.barplot(x='user', y='Unnamed: 0', hue='target', data=df3) 
# ax = plt.axes()
# ax.set_facecolor('black')
st.pyplot(fig)
st.write('Most of the users provide contributed to Car target class.')
st.write('Two categories of users 1-uses public transport, 2-not uses public transport')
st.markdown('1-u1, u3, u6, u7,`u9`,u12.')
st.markdown('2-u2, u4, u5, u6, u10,u11,u13')
st.write('users-u7, u11, u13 are the only ones not involved in either still/walking.')

##############
st.write('preprocessing')
st.write('Deleting unwanted columns')
df41 = pd.read_csv(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\EDA\corr.csv')
fig = plt.figure(figsize=(16, 8))
sns.heatmap(df41.corr(), vmin=-1, vmax=1, annot=True, cmap='coolwarm', annot_kws={'size':0})
st.pyplot(fig)
st.write('so, we can delete columns which are highly correlated to each other.')


################################
st.write('5.final dataset for training the model after preprocessing')
df21 = pd.read_csv(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Jupyter notebooks\final_fe.csv')
fig = msno.matrix(df21, sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0),inline=True)

st.pyplot(fig)
st.write('we replaced the nan values with mean of user of that particular sensor.')
st.write('There are no missing values in the dataset.')


########################
df61 = pd.read_csv(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Jupyter notebooks\lgbm.csv')
fig = px.bar(df61, x='Model', y='Accuracy', color = 'Model')
st.plotly_chart(fig)

#####################
#df61 = pd.read_csv(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Jupyter notebooks\lgbm.csv')
fig = px.bar(df61, x='Model', y='Time', color = 'Model')
st.plotly_chart(fig)