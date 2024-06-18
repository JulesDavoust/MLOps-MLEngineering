import joblib
import csv
import pandas as pd
import numpy as np

from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset, DataQualityTestPreset, DataStabilityTestPreset

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

import os
from dotenv import load_dotenv, set_key, find_dotenv
from pathlib import Path
import datetime

load_model = None
dir_path = 'reports'

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
version_model = os.getenv('version')
model_current = os.getenv('model')

def get_model():
    global load_model
    global model_current
    print("modeeel ",model_current)
    print(load_model)
    if load_model is None:
        print(model_current)
        load_model = joblib.load(f'models/{model_current}')
    return load_model

def predict(data_predict):
    global load_model
    try:
        load_model = get_model()
        print(data_predict)

        df = pd.read_csv('advertising_clean.csv')

        columns = [col for col in df.columns if col != 'Conversion_Rate']
        data_without_conversion_rate = df[columns]
        X = data_without_conversion_rate
        
        data = pd.DataFrame(data=np.array(data_predict), columns=['Campaign_Type', 'Target_Audience', 'Duration', 'Channel_Used', 'Location', 'Language', 'Clicks', 'Impressions', 'Engagement_Score', 'Customer_Segment'])
        categorical_columns = ['Campaign_Type', 'Target_Audience', 'Location', 'Language', 
                       'Customer_Segment', 'Channel_Used']
        data = pd.get_dummies(data, columns=categorical_columns)
        for col in X.columns:
            if col not in data.columns:
                data[col] = False
        data = data[X.columns]
        print(data)
        y_pred = load_model.predict(data)
        print(y_pred)
        data_in_csv(np.array(data), y_pred)
        
        return y_pred
    except Exception as e:
        return str(e)

def batch_predict(data_predict):
    try:
        load_model = get_model()
        print(data_predict.shape)
        df = pd.read_csv('advertising_clean.csv')

        columns = [col for col in df.columns if col != 'Conversion_Rate']
        data_without_conversion_rate = df[columns]
        X = data_without_conversion_rate
        
        data = pd.DataFrame(data=np.array(data_predict), columns=['Campaign_Type', 'Target_Audience', 'Duration', 'Channel_Used', 'Location', 'Language', 'Clicks', 'Impressions', 'Engagement_Score', 'Customer_Segment'])
        categorical_columns = ['Campaign_Type', 'Target_Audience', 'Location', 'Language', 'Customer_Segment', 'Channel_Used']
        data = pd.get_dummies(data, columns=categorical_columns)
        for col in X.columns:
            if col not in data.columns:
                data[col] = False
        data = data[X.columns]

        y_pred = load_model.predict(data)
        print(y_pred)
        data_in_csv(np.array(data), y_pred)
        return y_pred
    except Exception as e:
        return str(e)

def data_in_csv(data_predict, y):
        print('test ', y)
        data_list = data_predict.tolist()
        if data_predict.ndim > 1:
            with open('new_advertising.csv', 'a', newline='') as na:
                spamwriter = csv.writer(na, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                for i in range(0, int(data_predict.shape[0])):
                    data_list[i].append(y[i])
                    spamwriter.writerow(data_list[i])
        else:
            with open('new_advertising.csv', 'a', newline='') as na:
                spamwriter = csv.writer(na, delimiter=',', quoting=csv.QUOTE_MINIMAL)
                data_list.append(y)
                spamwriter.writerow(data_list)
        return True

def clean_dataset():
    try:
        df = pd.read_csv('new_advertising.csv')
        print(df)
        
        df = df.drop_duplicates(keep='first')
        df['Conversion_Rate'] = df['Conversion_Rate'].astype(float)
        df['Engagement_Score'] = df['Engagement_Score'].astype(float)
        df['Impressions'] = df['Impressions'].astype(int)
        df['Clicks'] = df['Clicks'].astype(int)
        df['Duration'] = df['Duration'].astype(int)
        df = df.dropna()
        
        iso_forst = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
        outliers = iso_forst.fit_predict(df)
        print(outliers)
        df['outliers'] = outliers
        print(df)
        df.to_csv('new_advertising.csv', index=False)
        return True
    except Exception as e:
        return str(e)

def run_data_drift_test_suite(reference: pd.DataFrame, current: pd.DataFrame):
    try:
        columns = [col for col in current.columns if (col != 'outliers')]
        current_without_conversion_rate = current[columns]
        data_drift_suite = TestSuite(tests=[DataDriftTestPreset()], timestamp=datetime.datetime.now() )
        data_drift_suite.run(reference_data=reference, current_data=current_without_conversion_rate)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        data_drift_suite.save_html(os.path.join(dir_path, "data_drift_suite.html"))
        return True
    except Exception as e:
        return str(e)
    
def result_data_drift_test_suite():
    try:
        result_path=os.path.join(dir_path, "data_drift_suite.html")
        with open(result_path, 'r', encoding='utf-8') as f:
            result_html = f.read()
        return result_html
    except Exception as e:
        return str(e)
    
def run_data_quality_test_suite(reference: pd.DataFrame, current: pd.DataFrame):
    try:
        columns = [col for col in current.columns if (col != 'outliers')]
        current_without_conversion_rate = current[columns]
        data_quality_suite = TestSuite(tests=[DataQualityTestPreset()], timestamp=datetime.datetime.now())
        data_quality_suite.run(reference_data=reference, current_data=current_without_conversion_rate)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        data_quality_suite.save_html(os.path.join(dir_path, "data_quality_suite.html"))
        return True
    except Exception as e:
        return str(e)
    
def result_data_quality_test_suite():
    try:
        result_path=os.path.join(dir_path, "data_quality_suite.html")
        with open(result_path, 'r', encoding='utf-8') as f:
            result_html = f.read()
        return result_html
    except Exception as e:
        return str(e)
    
def run_data_stability_test_suite(reference: pd.DataFrame, current: pd.DataFrame):
    try:
        columns = [col for col in current.columns if (col != 'outliers')]
        current_without_conversion_rate = current[columns]
        data_stability_suite = TestSuite(tests=[DataStabilityTestPreset()], timestamp=datetime.datetime.now())
        data_stability_suite.run(reference_data=reference, current_data=current_without_conversion_rate)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        data_stability_suite.save_html(os.path.join(dir_path, "data_stability_suite.html"))
        return True
    except Exception as e:
        return str(e)
    
def result_data_stability_test_suite():
    try:
        result_path=os.path.join(dir_path, "data_stability_suite.html")
        with open(result_path, 'r', encoding='utf-8') as f:
            result_html = f.read()
        return result_html
    except Exception as e:
        return str(e)
    
def MSE():
    try:
        df_true = pd.read_csv('advertising_clean.csv')
        df_pred = pd.read_csv('new_advertising.csv')
        if len(df_true['Conversion_Rate']) >= len(df_pred['Conversion_Rate']):
            return mean_squared_error(df_true['Conversion_Rate'].head(len(df_pred['Conversion_Rate'])), df_pred['Conversion_Rate'].head(len(df_pred['Conversion_Rate'])))
        else:
            return mean_squared_error(df_true['Conversion_Rate'].head(len(df_true['Conversion_Rate'])), df_pred['Conversion_Rate'].head(len(df_true['Conversion_Rate'])))
    except Exception as e:
        return str(e)
    
def MAE():
    try:
        df_true = pd.read_csv('advertising_clean.csv')
        df_pred = pd.read_csv('new_advertising.csv')
        if len(df_true['Conversion_Rate']) >= len(df_pred['Conversion_Rate']):
            return mean_absolute_error(df_true['Conversion_Rate'].head(len(df_pred['Conversion_Rate'])), df_pred['Conversion_Rate'].head(len(df_pred['Conversion_Rate'])))
        else:
            return mean_absolute_error(df_true['Conversion_Rate'].head(len(df_true['Conversion_Rate'])), df_pred['Conversion_Rate'].head(len(df_true['Conversion_Rate'])))
    except Exception as e:
        return str(e)
    
def R2():
    try:
        df_true = pd.read_csv('advertising_clean.csv')
        df_pred = pd.read_csv('new_advertising.csv')
        if len(df_true['Conversion_Rate']) >= len(df_pred['Conversion_Rate']):
            return r2_score(df_true['Conversion_Rate'].head(len(df_pred['Conversion_Rate'])), df_pred['Conversion_Rate'].head(len(df_pred['Conversion_Rate'])))
        else:
            return r2_score(df_true['Conversion_Rate'].head(len(df_true['Conversion_Rate'])), df_pred['Conversion_Rate'].head(len(df_true['Conversion_Rate'])))
    except Exception as e:
        return str(e)
    
def history():
    try:
        df_pred = pd.read_csv('new_advertising.csv')
        return df_pred
    except Exception as e:
        return str(e)
    
def retrain():
    global load_model
    try:
        load_model = get_model()
        df_reference = pd.read_csv('advertising_clean.csv')
        df_current = pd.read_csv('new_advertising.csv')
        df_new = pd.concat([df_reference, df_current], ignore_index=True, sort=False)
        print(df_new)

        columns = [col for col in df_new.columns if (col != 'Conversion_Rate' and col != 'outliers')]
        data_without_conversion_rate = df_new[columns]

        X = data_without_conversion_rate
        y = df_new['Conversion_Rate']

        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model = LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        y_pred_old = load_model.predict(X_test)

        df = verif_model(y_pred_old, y_pred, y_test)

        now = datetime.datetime.now()
        dt_string = now.strftime("%dd%m%Y_%Hh%Mm%Ss")
        print(dt_string)

        joblib.dump(model, f'models/model_{dt_string}.pkl')
        inc_version()

        return df
    except Exception as e:
        print(e)
        return pd.DataFrame()
    
def verif_model(y_pred_old, y_pred_new, y_true):

    mse_new = mean_squared_error(y_true, y_pred_new)
    mse_old = mean_squared_error(y_true, y_pred_old)

    mae_new = mean_absolute_error(y_true, y_pred_new)
    mae_old = mean_absolute_error(y_true, y_pred_old)

    r2_old = r2_score(y_true, y_pred_old)
    r2_new = r2_score(y_true, y_pred_new)
    d = {'Current':np.array([mse_new, mae_new, r2_new]), 'New':np.array([mse_old, mae_old, r2_old])}
    df = pd.DataFrame(data=d, index=['mse', 'mae', 'r2'])
    print(df)
    return df

def view_models():
    files = {os.listdir('models').index(f):f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))}
    df = pd.DataFrame(data=files, index=['models'])
    return df

def json_models():
    files = {os.listdir('models').index(f):f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))}
    df = pd.DataFrame(data=files, index=['models'])
    return df

def select_models(index):
    global load_model
    global model_current
    files = {os.listdir('models').index(f):f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))}
    set_key(dotenv_path, 'model', files[index])
    model_current = files[index]
    print(model_current)
    load_model = None

def delete_model(index):
    global model_current
    global load_model
    if(index == 0):
        return 'Impossible'
    else:
        files = {os.listdir('models').index(f):f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))}
        if(files[index] == model_current):
            set_key(dotenv_path, 'model', 'model.pkl')
            model_current = 'model.pkl'
        os.remove(f'models/{files[index]}')
        load_model = None
        return 'ok'
    
def current_model():
    global model_current
    return model_current

def indice_model(index):
    files = {os.listdir('models').index(f):f for f in os.listdir('models') if os.path.isfile(os.path.join('models', f))}
    i_model = joblib.load(f'models/{files[index]}')

    df_reference = pd.read_csv('advertising_clean.csv')
    df_current = pd.read_csv('new_advertising.csv')
    df_new = pd.concat([df_reference, df_current], ignore_index=True, sort=False)

    columns = [col for col in df_new.columns if (col != 'Conversion_Rate' and col != 'outliers')]
    data_without_conversion_rate = df_new[columns]

    X = data_without_conversion_rate
    y = df_new['Conversion_Rate']

    X_train, X_test, y_train, y_test = train_test_split(X, y)
    y_pred = i_model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)

    mae = mean_absolute_error(y_test, y_pred)

    r2 = r2_score(y_test, y_pred)

    d = {'Indicators':np.array([mse, mae, r2])}
    df = pd.DataFrame(data=d, index=['mse', 'mae', 'r2'])
    return df
    
    
def inc_version():
    global version_model
    version_model_tab = version_model.split('.')
    if(int(version_model_tab[2]) < 9):
        version_model_tab[2] = str(int(version_model_tab[2]) + 1)
    elif(int(version_model_tab[1]) < 9):
        version_model_tab[2] = str(0)
        version_model_tab[1] = str(int(version_model_tab[1]) + 1)
    else:
        version_model_tab[1] = str(0)
        version_model_tab[0] = str(int(version_model_tab[0]) + 1)
    version_model = version_model_tab[0]+'.'+version_model_tab[1]+'.'+version_model_tab[2]
    print(version_model)
    set_key(dotenv_path, 'version', version_model)
    
def version():
    global version_model
    print(version_model)
    return version_model

def info_model_coef():
    model = get_model()
    return model.coef_

def info_model_intercept():
    model = get_model()
    return model.intercept_

def give_data(data):
    try:
        tab = []
        for i in range(0, len(data[0])):
            tab.append(data[0][i])
        tab = np.array(tab)
        print('tab ',tab)
        df = pd.read_csv('advertising_clean.csv')

        columns = [col for col in df.columns if col != 'Conversion_Rate']
        data_without_conversion_rate = df[columns]
        X = data_without_conversion_rate
        
        data = pd.DataFrame(data=np.array([tab[0:(len(tab)-1)]]), columns=['Campaign_Type', 'Target_Audience', 'Duration', 'Channel_Used', 'Location', 'Language', 'Clicks', 'Impressions', 'Engagement_Score', 'Customer_Segment'])
        categorical_columns = ['Campaign_Type', 'Target_Audience', 'Location', 'Language', 
                       'Customer_Segment', 'Channel_Used']
        data = pd.get_dummies(data, columns=categorical_columns)
        for col in X.columns:
            if col not in data.columns:
                data[col] = False
        data = data[X.columns]
        print('test :',np.array(data), np.array([tab[len(tab)-1]]))
        return data_in_csv(np.array(data), np.array([tab[len(tab)-1]]))
    except Exception as e:
        return e
    
def aberant_values():
    df_current = pd.read_csv('new_advertising.csv')
    df = df_current[df_current['outliers'] == -1]
    return df

def json_aberant_values():
    df_current = pd.read_csv('new_advertising.csv')
    df = df_current[df_current['outliers'] == -1]
    d = df.to_dict()
    return d


def del_aberant_values():
    df_current = pd.read_csv('new_advertising.csv')
    df = df_current[df_current['outliers'] != -1]
    df.to_csv('new_advertising.csv', index=False)
    return True

def correct_value_y(value: int, row: int):
    df_current = pd.read_csv('new_advertising.csv')
    print(value)
    print(df_current.shape[1])
    df_current['Conversion_Rate'][row] = value
    print(df_current)
    df_current.to_csv('new_advertising.csv', index=False)
    return True