import data_handler as dh
import data_train as trm
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.experimental  import enable_hist_gradient_boosting # Necesary for HistGradientBoostingClassifier
from sklearn.ensemble      import HistGradientBoostingClassifier
from xgboost               import XGBClassifier
from lightgbm              import LGBMClassifier
from catboost              import CatBoostClassifier


oet, scaler, knn, x_train, x_test, y_train, y_test = dh.get_data(r'C:\Users\rnr31\Documents\GitHub\apple_watch_tracker\Datasets\dataset_0.5sec.csv', 26000)
hist_gbm, xgb, lgbm, cb = trm.train_model(x_train, y_train)


def test_model(x_test, y_test):
    hist_gbm_pred = hist_gbm.predict(x_test)

    xgb_pred = xgb.predict(x_test)

    lgbm_pred = lgbm.predict(x_test)

    cb_pred = cb.predict(x_test)

    hist_score = accuracy_score(y_test, hist_gbm_pred)
    xgb_score = accuracy_score(y_test, xgb_pred)
    lgbm_score = accuracy_score(y_test, lgbm_pred)
    cb_score = accuracy_score(y_test, cb_pred)

    return hist_score, xgb_score, lgbm_score, cb_score

hist_score, xgb_score, lgbm_score, cb_score = test_model(x_test,y_test)
print(hist_score, xgb_score, lgbm_score, cb_score)