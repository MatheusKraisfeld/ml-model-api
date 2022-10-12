import os
import pickle

import mlflow.sklearn
import pandas as pd

df = mlflow.search_runs(
    experiment_ids="0",
    filter_string="",
    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    max_results=1,
    order_by=["metrics.rmse ASC"],
)

run_id = df.loc[df["metrics.rmse"].idxmin()]["run_id"]

model = mlflow.sklearn.load_model("runs:/" + run_id + "/model")

with open("../api/best_model.pkl", "wb") as f:
    pickle.dump(model, f)

data = pd.read_csv("~/ml-model-api/sources/pandas_df_success.csv")
test = data.drop(["class", "classEncoder"], axis=1)

print(run_id)
print(model.predict(test))
