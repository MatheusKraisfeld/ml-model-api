import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn import svm
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    try:
        data = pd.read_csv("~/ml-model-api/sources/pandas_df_success.csv")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    data = data.drop(["class"], axis=1)
    train, test = train_test_split(data)

    train_x = train.drop(["classEncoder"], axis=1)
    test_x = test.drop(["classEncoder"], axis=1)
    train_y = train[["classEncoder"]]
    test_y = test[["classEncoder"]]

    with mlflow.start_run(experiment_id=0):
        random = 45
        
        clf = svm.LinearSVC(C=1.0, max_iter=1000, verbose=0)
        
        clf.fit(train_x, train_y)

        predicted_qualities = clf.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("random-state", random)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(clf, "model", registered_model_name="clf_iris_model_2")
        else:
            mlflow.sklearn.log_model(clf, "model")