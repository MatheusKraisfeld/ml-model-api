import pandas as pd
from io import StringIO
from pandas_schema import Column, Schema
from pandas_schema.validation import LeadingWhitespaceValidation, TrailingWhitespaceValidation, MatchesPatternValidation, InRangeValidation, InListValidation

pd_df_iris = pd.read_csv('~/ml-model-api/sources/iris.txt')

# min: inclusive
# max: exclusive
schema = Schema([
    Column('sepal_length', [InRangeValidation(4.3, 7.91)]),
    Column('sepal_width', [InRangeValidation(2.0, 4.41)]),
    Column('petal_length', [InRangeValidation(1.0, 6.91)]),
    Column('petal_width', [InRangeValidation(0.1, 2.51)]),
    Column('classEncoder', [InRangeValidation(0, 3)]),
    Column('class', [InListValidation(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])])
])

errors = schema.validate(pd_df_iris)

for error in errors:
    print(error)

filtered_df = pd_df_iris.drop(pd_df_iris.index[[5, 15, 22, 25, 28]])

filtered_df.to_csv('~/ml-model-api/sources/pandas_df_success.csv')

fail_df = pd_df_iris.iloc[[5, 15, 22, 25, 28]]

fail_df['messageError'] = ""

i = 0
for error in errors:
  fail_df.loc[error.row:error.row, 'messageError'] = str(error.column) + " " + str(error.message)
  i = i + 1

fail_df.to_csv('~/ml-model-api/sources/pandas_df_fail.csv')