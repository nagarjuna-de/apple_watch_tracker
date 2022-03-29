#### Importing libraries
import numpy    as np
from numpy.testing._private.utils import decorate_methods
import pandas   as pd
import seaborn  as sb
import matplotlib.pyplot as plt
import sklearn  as skl
import time

from sklearn import pipeline      # Pipeline
from sklearn import preprocessing # OrdinalEncoder, LabelEncoder
from sklearn import impute
from sklearn import compose
from sklearn import model_selection # train_test_split
from sklearn import metrics         # accuracy_score, balanced_accuracy_score, plot_confusion_matrix
from sklearn import set_config

from sklearn.tree          import DecisionTreeRegressor
from sklearn.ensemble      import RandomForestRegressor
from sklearn.ensemble      import ExtraTreesRegressor
from sklearn.ensemble      import AdaBoostRegressor
from sklearn.ensemble      import GradientBoostingRegressor
from xgboost               import XGBRegressor
from lightgbm              import LGBMRegressor
from catboost              import CatBoostRegressor

#### Reading Data from CSV file.
data = pd.read_csv(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\dataset_5sec.csv')

#### Dropping not required columns.
data.drop(['Unnamed: 0','id', 'activityrecognition#0','user'],axis=1, inplace=True)

#### Renaming Columns
data_col = ['time','activityrecognition_1']
for i in data.columns[2:58]:
    b = i.split('.')[2].split('#')
    data_col.append(f'{b[0]}_{b[1]}')

for i in data.columns[58:66]:
    b = i.split('#')
    data_col.append(f'{b[0]}_{b[1]}')

data_col.append('target')

data = pd.DataFrame.from_records(data.values)
data.columns = data_col

#### Creating features(x) & target (y)

x = data.drop(['target'], axis=1)
y = data['target']

