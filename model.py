import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest

def clean_nanValue(df):
    return df.dropna()

df = pd.read_csv('advertising.csv')
df = df.dropna()
df = df.drop_duplicates()
df = df.astype(float)
# iso_forst = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
# outliers = iso_forst.fit_predict(df)
# print(outliers)
# df['outliers'] = outliers
# df.info()
# print(df.head())
# df_clean = clean_nanValue(df)
# df.to_csv('advertising_clean.csv', index=False)

X = df[['TV', 'radio', 'newspaper']]
y = df['sales']

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

y_pred = model.predict([[234.1, 9, 10]])
print(y_pred)

joblib.dump(model, 'model.pkl')