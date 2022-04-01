import time
from IPython.display import clear_output
import numpy    as np
import pandas   as pd
import seaborn  as sns
sns.set_style("whitegrid")
import matplotlib.pyplot as plt
import sklearn  as skl
from numpy import asarray

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline      
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


###################



def get_data(pth, x):
    data = pd.read_csv(pth)
    oe = OrdinalEncoder()
    cu = asarray(data['user'])
    data['user'] = oe.fit_transform(cu.reshape(-1,1))
    data = data.sort_values(by='user')
    oet = OrdinalEncoder()
    ct = asarray(data['target'])
    data['target'] = oet.fit_transform(ct.reshape(-1,1))

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

        data_col.append('target')
        data_col.append('user')
        data = pd.DataFrame.from_records(data.values)
        data.columns = data_col

        return data

    data = col_names_change(data)

    ##### Dropping columns
    data.drop(['pressure_mean', 'pressure_max', 'pressure_min', 'pressure_std', 'sound_std', 'speed_std','step_counter_mean','step_counter_std','light_std',
            'proximity_mean','proximity_min','proximity_max','proximity_std', 'step_counter_min', 'step_counter_max' ], axis=1, inplace=True)

    df0 = data[data['target']==0]
    df1 = data[data['target']==1]
    df2 = data[data['target']==2]
    df3 = data[data['target']==3]
    df4 = data[data['target']==4]

    def fill_nan_values(tsdf):
        for i in tsdf.columns:
            ndf = tsdf.groupby(['target'])[f'{i}'].mean()
            mn = ndf.values[0]
            tsdf[f'{i}'] = tsdf[f'{i}'].fillna(mn)
        return tsdf

    df0 = fill_nan_values(df0)
    df1 = fill_nan_values(df1)
    df2 = fill_nan_values(df2)
    df3 = fill_nan_values(df3)
    df4 = fill_nan_values(df4)

    data = pd.concat([df0, df1, df2, df3, df4])
    data = data.sort_values(by='user')
    data.drop(['user'], axis=1, inplace=True)
    ### deleting min_max col:
    def del_min_max(data):
        accelerometer_min = asarray(data['accelerometer_min'])
        accelerometer_max = asarray(data['accelerometer_max'])
        gyroscope_uncalibrated_min = asarray(data['gyroscope_uncalibrated_min'])
        gyroscope_uncalibrated_max = asarray(data['gyroscope_uncalibrated_max'])
        linear_acceleration_min = asarray(data['linear_acceleration_min'])
        linear_acceleration_max = asarray(data['linear_acceleration_max'])
        ######
        data = data.drop(data.filter(regex='min').columns, axis=1)
        data = data.drop(data.filter(regex='max').columns, axis=1)
        #####
        data['accelerometer_min'] = accelerometer_min 
        data['accelerometer_max'] = accelerometer_max
        data['gyroscope_uncalibrated_min'] = gyroscope_uncalibrated_min
        data['gyroscope_uncalibrated_max'] = gyroscope_uncalibrated_max
        data['linear_acceleration_min'] = linear_acceleration_min
        data['linear_acceleration_max'] = linear_acceleration_max

        return data
    data = del_min_max(data)

    data_test = data.iloc[:x, :]
    data_train = data.iloc[x: , :]

    #### Training sets
    x_train = data_train.drop(['target'], axis=1)
    y_train = data_train['target']

    #### Testing sets
    x_test = data_test.drop(['target'], axis=1)
    y_test = data_test['target']

    ### Scaler
    scaler = StandardScaler()
    scaler = scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    ### Imputer
    knn = KNNImputer()
    knn = knn.fit(x_train)

    return oet, scaler, knn, x_train, x_test, y_train, y_test




