import data_handler as dh
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier

oet, scaler, knn, x_train, x_test, y_train, y_test = dh.get_data(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Datasets\dataset_0.5sec.csv', 26000)

def train_model(x_train, y_train):
    # hist_gbm = HistGradientBoostingClassifier(max_iter=600, max_depth=3, max_leaf_nodes=6, min_samples_leaf=5)
    # hist_gbm = hist_gbm.fit(x_train,y_train)


    # xgb = XGBClassifier(n_estimators=100)
    # xgb = xgb.fit(x_train,y_train)

    lgbm = LGBMClassifier(n_estimators=165, num_leaves=45, max_depth=10, boosting_type='goss', learning_rate=0.15)
    lgbm = lgbm.fit(x_train,y_train)


    # cb = CatBoostClassifier(n_estimators=100)
    # cb = cb.fit(x_train,y_train)

    return  lgbm