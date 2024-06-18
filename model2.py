import pandas as pd

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Charger le dataset
data = pd.read_csv('marketing_campaign_dataset.csv')

# Afficher les premières lignes du dataset pour visualiser les données
print(data.head())

# Conversion du type de données
data['Conversion_Rate'] = data['Conversion_Rate'].astype(float)
data['Duration'] = data['Duration'].apply(lambda x: int(x.split()[0])) 
# data['Date'] = pd.to_datetime(data['Date'])  # Conversion en format date

# Supprimer le symbole de devise et convertir en float
data['Acquisition_Cost'] = data['Acquisition_Cost'].replace('[\$,]', '', regex=True).astype(float)

# Remplir ou supprimer les valeurs manquantes (NaN)
# Par exemple, si une colonne a des valeurs manquantes, on peut utiliser la moyenne pour les remplir
data['Conversion_Rate'].fillna(data['Conversion_Rate'].mean(), inplace=True)


data['Acquisition_Cost'].fillna(data['Acquisition_Cost'].mean(), inplace=True)

# Supprimer les lignes ou les colonnes entièrement manquantes (optionnel)
data.dropna(how='all', inplace=True)  # Supprimer les lignes complètement manquantes
data.dropna(axis=1, how='all', inplace=True)  # Supprimer les colonnes complètement manquantes

# Normaliser les textes (par exemple, pour la colonne 'Language')
data['Language'] = data['Language'].str.lower() 


columns = [col for col in data.columns if col != 'Conversion_Rate']
columns.append('Conversion_Rate')
data = data[columns]

# Vérifier les modifications
print(data.head())

# Sauvegarder le dataset nettoyé
data.to_csv('cleaned_campaign_data.csv', index=False)





data = pd.read_csv('cleaned_campaign_data.csv')


features = ['Campaign_Type', 'Target_Audience', 'Duration', 'Channel_Used',
            'Location', 'Language', 'Clicks', 'Impressions', 'Engagement_Score', 
            'Customer_Segment']

data_filtered = data[features + ['Conversion_Rate']]

categorical_columns = ['Campaign_Type', 'Target_Audience', 'Location', 'Language', 
                       'Customer_Segment', 'Channel_Used']

data_filtered = pd.get_dummies(data_filtered, columns=categorical_columns)

# Convertir la colonne 'Date' en caractéristiques temporelles supplémentaires
# data_filtered['Year'] = pd.to_datetime(data_filtered['Date']).dt.year
# data_filtered['Month'] = pd.to_datetime(data_filtered['Date']).dt.month
# data_filtered['Day'] = pd.to_datetime(data_filtered['Date']).dt.day
# data_filtered.drop(columns=['Date'], inplace=True)

# Vérifier les modifications
print(data_filtered.head())

columns = [col for col in data_filtered.columns if col != 'Conversion_Rate']

# Ajouter 'Conversion_Rate' à la fin
columns.append('Conversion_Rate')

# Réorganiser les colonnes du DataFrame
data_filtered = data_filtered[columns]

# Sauvegarder le dataset transformé
data_filtered.to_csv('transformed_campaign_data.csv', index=False)

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import IsolationForest


df = pd.read_csv('transformed_campaign_data.csv')

columns = [col for col in df.columns if col != 'Conversion_Rate']
data_without_conversion_rate = df[columns]

X = data_without_conversion_rate
y = df['Conversion_Rate']

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression(fit_intercept=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred)

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# y_pred = model.predict([[234.1, 9, 10]])
# print(y_pred)

example_data = {
    'Company': ['Innovate Industries'],
    'Campaign_Type': ['Display'],
    'Target_Audience': ['Women 35-44'],
    'Duration': [30],
    'Channel_Used': ['Google Ads'],
    'Location': ['Chicago'],
    'Language': ['Spanish'],
    'Clicks': [506],
    'Impressions': [1922],
    'Engagement_Score': [6],
    'Customer_Segment': ['Health & Wellness']
}

example_df = pd.DataFrame(example_data)

# Appliquer les mêmes transformations que celles appliquées au dataset d'entraînement

# Colonnes catégorielles à convertir en one-hot encoding
categorical_columns = ['Company', 'Campaign_Type', 'Target_Audience', 'Location', 'Language', 
                       'Customer_Segment', 'Channel_Used']

# Appliquer one-hot encoding
example_df = pd.get_dummies(example_df, columns=categorical_columns)

# Convertir la colonne 'Date' en caractéristiques temporelles supplémentaires
# example_df['Year'] = pd.to_datetime(example_df['Date']).dt.year
# example_df['Month'] = pd.to_datetime(example_df['Date']).dt.month
# example_df['Day'] = pd.to_datetime(example_df['Date']).dt.day
# example_df.drop(columns=['Date'], inplace=True)

# Ajouter les colonnes manquantes avec des valeurs par défaut (0)
# Cela est nécessaire pour s'assurer que l'exemple a les mêmes colonnes que les données d'entraînement
for col in X.columns:
    if col not in example_df.columns:
        example_df[col] = 0

# Réordonner les colonnes pour correspondre à l'ordre des données d'entraînement
example_df = example_df[X.columns]

# Afficher les données de l'exemple
print(example_df)


y_pred = model.predict(example_df)
print(y_pred)

joblib.dump(model, 'model_v2.pkl')