import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import requests
import html


global url_api
url_api = 'http://localhost:5000'

if 'page' not in st.session_state:
    st.session_state['page'] = 'Home'

sb = st.sidebar
varHome = sb.button("Home")
varGiveData = sb.button("Give data")
varPred = sb.button("Predictions")
varHistory = sb.button("View History")
varRetrain = sb.button("Retrain model")
varCorrectValue = sb.button("Correct Y value")
varCurrentModel = sb.button("Current model")
csb_info = sb.container(height= 230)
csb_info.write('Informations about the model')
csb_c = csb_info.button('Coefficients')
csb_i = csb_info.button('Intercept')
csb_im = csb_info.button('Indicators about the model')
varViewModels = sb.button("Select model")
varDeleteModels = sb.button("Delete model")
varCleanDataSet = sb.button("Clean dataset")
varAberantValue = sb.button("Aberant values")
csb = sb.container(height= 400)
csb.write('Model monitoring')
csb_dd = csb.button('Data drift')
csb_dq = csb.button('Data quality')
csb_ds = csb.button('Data stability')
csb_mse = csb.button('Data MSE')
csb_mae = csb.button('Data MAE')
csb_r2 = csb.button('Data R2')

if(varHome):
    st.session_state['page']='Home'
elif(varPred):
    st.session_state['page']='Predictions'
elif(varHistory):
    st.session_state['page']='ViewHistory'
elif(varViewModels):
    st.session_state['page']='ViewModels'
elif(varCurrentModel):
    st.session_state['page']='CurrentModel'
elif(varDeleteModels):
    st.session_state['page']='DeleteModel'
elif(varCleanDataSet):
    st.session_state['page']='CleanDataSet'
elif(csb_dd):
    st.session_state['page']='DataDrift'
elif(csb_dq):
    st.session_state['page']='DataQuality'
elif(csb_ds):
    st.session_state['page']='DataStability'
elif(csb_mse):
    st.session_state['page']='MSE'
elif(csb_mae):
    st.session_state['page']='MAE'
elif(csb_r2):
    st.session_state['page']='R2'
elif(csb_c):
    st.session_state['page']='Coefficient'
elif(csb_i):
    st.session_state['page']='Intercept'
elif(csb_im):
    st.session_state['page']='Indicators'
elif(varAberantValue):
    st.session_state['page']='AberantValues'
elif(varGiveData):
    st.session_state['page']='GiveData'
elif(varCorrectValue):
    st.session_state['page']='CorrectValue'
elif(varRetrain):
    st.session_state['page']='Retrain'


def display_model():
    return requests.get(url=url_api+'/json/models').json()

def current_model():
    return requests.get(url=url_api+'/current-model').json()


def MSE():
    return requests.get(url=url_api+'/monitoring/MSE').json()


def MAE():
    return requests.get(url=url_api+'/monitoring/MAE').json()


def R2():
    return requests.get(url=url_api+'/monitoring/R2').json()


def datadrif():
    data = requests.post(url=url_api+'/monitoring/process/data-drift').json()
    return st.write(f"{data['message']} you can display it.")


def dataquality():
    data = requests.post(url=url_api+'/monitoring/process/data-quality').json()
    return st.write(f"{data['message']} you can display it.")


def datastability():
    data = requests.post(url=url_api+'/monitoring/process/data-stability').json()
    return st.write(f"{data['message']} you can display it.")

if(st.session_state['page']=='Home'):
    data = requests.get(url_api).json()
    version = requests.get(url_api+'/version-model').json()
    st.title(data)
    st.write(f"Version : {version['version']}")


if(st.session_state['page']=='GiveData'):
    st.title('Give Data')
    value = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    df = pd.read_csv('cleaned_campaign_data.csv')

    var1 = df['Campaign_Type'].drop_duplicates(keep='first')
    var2 = df['Target_Audience'].drop_duplicates(keep='first')
    var3 = df['Duration'].drop_duplicates(keep='first')
    var4 = df['Channel_Used'].drop_duplicates(keep='first')
    var5 = df['Location'].drop_duplicates(keep='first')
    var6 = df['Language'].drop_duplicates(keep='first')
    var7 = df['Clicks'].drop_duplicates(keep='first')
    var8 = df['Impressions'].drop_duplicates(keep='first')
    var9 = df['Engagement_Score'].drop_duplicates(keep='first')
    var10 = df['Customer_Segment'].drop_duplicates(keep='first')


    st.write('Enter your values that you want to give :')
    ct = st.text_input(label='Campaign Type')
    ta =  st.text_input(label='Target Audience')
    duration = st.text_input(label='Duration')
    cu = st.text_input(label='Channel Used')
    location = st.text_input(label='Location')
    language = st.text_input(label='Language')
    clicks = st.text_input(label='Clicks')
    impressions = st.text_input(label='Impressions')
    es = st.text_input(label='Engagement Score')
    cs = st.text_input(label='Customer Segment')
    conversion_rate = st.text_input(label='Conversion Rate')
    button_send = st.button('Send values')

    if(button_send):
        value[0] = ct
        value[1] = ta
        value[2] = int(duration)
        value[3] = cu
        value[4] = location
        value[5] = language.lower()
        value[6] = int(clicks)
        value[7] = int(impressions)
        value[8] = int(es)
        value[9] = cs
        value[10] = float(conversion_rate)

        df.loc[len(df)] = {
            'Campaign_ID':None,
            'Company':'',
            'Campaign_Type': ct,
            'Target_Audience': ta,
            'Duration': int(duration),
            'Channel_Used': cu,
            'Acquisition_Cost':None,
            'ROI':None,
            'Location': location,
            'Language': language.lower(),
            'Clicks': int(clicks),
            'Impressions': int(impressions),
            'Engagement_Score': int(es),
            'Customer_Segment': cs,
            'Conversion_Rate': float(conversion_rate)
        }
        
        data = requests.post(url=url_api+'/give-data', json={'data':value}).json()
        df.to_csv('cleaned_campaign_data.csv', index=False)
        print(data)
        if(data['data-gived']):
            st.write('Values added')
        else:
            st.write('Error')

    
if(st.session_state['page']=='Predictions'):
    st.title('Predictions')
    predict = st.selectbox('Predict :', ('One value', 'Multiple values'), placeholder='Choose an option')

    if(predict == 'One value'):
        value = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        st.write('Enter your values')
        df = pd.read_csv('cleaned_campaign_data.csv')

        var1 = df['Campaign_Type'].drop_duplicates(keep='first')
        var2 = df['Target_Audience'].drop_duplicates(keep='first')
        var3 = df['Duration'].drop_duplicates(keep='first')
        var4 = df['Channel_Used'].drop_duplicates(keep='first')
        var5 = df['Location'].drop_duplicates(keep='first')
        var6 = df['Language'].drop_duplicates(keep='first')
        var7 = df['Clicks'].drop_duplicates(keep='first')
        var8 = df['Impressions'].drop_duplicates(keep='first')
        var9 = df['Engagement_Score'].drop_duplicates(keep='first')
        var10 = df['Customer_Segment'].drop_duplicates(keep='first')


        st.write('Enter your values that you want to give :')
        ct = st.selectbox(label='Campaign Type', options=np.array(var1))
        ta =  st.selectbox(label='Target Audience', options=np.array(var2))
        duration = st.text_input(label='Duration', value=0)
        cu = st.selectbox(label='Channel Used', options=np.array(var4))
        location = st.selectbox(label='Location', options=np.array(var5))
        language = st.selectbox(label='Language', options=np.array(var6))
        clicks = st.text_input(label='Clicks', value=0)
        impressions = st.text_input(label='Impressions', value=0)
        es = st.text_input(label='Engagement Score', value=0)
        cs = st.selectbox(label='Customer Segment', options=np.array(var10))
        button_send = st.button('Send values')

        st.write('Value predict :')
        if(button_send):
            value[0] = ct
            value[1] = ta
            value[2] = int(duration)
            value[3] = cu
            value[4] = location
            value[5] = language
            value[6] = int(clicks)
            value[7] = int(impressions)
            value[8] = int(es)
            value[9] = cs
            data = requests.post(url=url_api+'/prevision/predict', json={'features':value}).json()
            print(data['y_pred'])
            st.write(data['y_pred'][0])

        print(value)

    elif(predict == 'Multiple values'):
        value = []
        nombre_pred = st.number_input('How many predictions do you want ?', min_value=1)
        st.write('Enter your values (separate it with "," like this : 183.2,28) :')

        df = pd.read_csv('cleaned_campaign_data.csv')

        var1 = df['Campaign_Type'].drop_duplicates(keep='first')
        var2 = df['Target_Audience'].drop_duplicates(keep='first')
        var3 = df['Duration'].drop_duplicates(keep='first')
        var4 = df['Channel_Used'].drop_duplicates(keep='first')
        var5 = df['Location'].drop_duplicates(keep='first')
        var6 = df['Language'].drop_duplicates(keep='first')
        var7 = df['Clicks'].drop_duplicates(keep='first')
        var8 = df['Impressions'].drop_duplicates(keep='first')
        var9 = df['Engagement_Score'].drop_duplicates(keep='first')
        var10 = df['Customer_Segment'].drop_duplicates(keep='first')


        st.write('Enter your values that you want to give :')
        ct = st.selectbox(label='Campaign Type', options=np.array(var1))
        ta =  st.selectbox(label='Target Audience', options=np.array(var2))
        duration = st.text_input(label='Duration', value=0)
        cu = st.selectbox(label='Channel Used', options=np.array(var4))
        location = st.selectbox(label='Location', options=np.array(var5))
        language = st.selectbox(label='Language', options=np.array(var6))
        clicks = st.text_input(label='Clicks', value=0)
        impressions = st.text_input(label='Impressions', value=0)
        es = st.text_input(label='Engagement Score', value=0)
        cs = st.selectbox(label='Customer Segment', options=np.array(var10))

        add = st.button('Add value')
        button_send = st.button('Send to model')
        st.write('Values predict :')
        if(add):
            val = []
            val.append(ct)
            val.append(ta)
            val.append(int(duration))
            val.append(cu)
            val.append(location)
            val.append(language)
            val.append(int(clicks))
            val.append(int(impressions))
            val.append(int(es))
            val.append(int(cs))
            value.append(val)
            print(value)
        if(button_send):
            print(value)
            data = requests.post(url=url_api+'/prevision/batch-predict', json={'features':value}).json()
            print(len(data['y_pred']))
            for j in range(0, len(data['y_pred'])):
                print(j)
                st.write(f'{j+1} value predict : ',data['y_pred'][j])


if(st.session_state['page']=='Retrain'):
    st.title('Retrain the model')
    st.write('To retrain it you have to click on the button. Then it will return to you the indicator between the current model (the model that you use) and the new model (the model trained). To choose one of these models you have to go in "Select model".'  )
    retrain = st.button('Retrain')
    if(retrain):
        data = requests.get(url=url_api+'/retrain-model').json()
        print(data)
        data = data['verif']
        df = pd.DataFrame(data)
        
        print(df)
        st.write(df)

if(st.session_state['page']=='Coefficient'):
    coef = requests.get(url=url_api+'/informations-model/coef').json()
    st.title('Coefficient')
    st.write("Here it is the coefficients that use the current model :")
    for i in range(0, len(coef['coeffecients'])):
        st.write(f"{i+1} coefficient : {coef['coeffecients'][i]}")

if(st.session_state['page']=='Intercept'):
    intercept = requests.get(url=url_api+'/informations-model/intercept').json()
    st.title('Intercept')
    st.write("Here it is the intercept of the current model :")
    st.write(f"Intercept : {intercept['intercept']}")

if(st.session_state['page']=='Indicators'):
    st.title('Indicators')
    st.write('Choose a model to see its indicators')
    data = display_model()
    df = pd.DataFrame(data=data["models"])
    st.write(df)
    print(np.array(df).shape)
    num_model = st.number_input('Choose the model :', min_value=0,
                    max_value=(np.array(df).shape)[1]-1)
    choose_model = st.button('Choose')
    st.write("Here it is the indicators about the model (so the indicators that show the performance of the model) :")
    if(choose_model):
        indicators = requests.get(url=url_api+'/indicator-model', json={'index':num_model}).json()
        st.write(f"Mean Squared Error : {indicators['Indicators']['Indicators']['mse']}")
        st.write(f"Mean Absolute Error : {indicators['Indicators']['Indicators']['mae']}")
        st.write(f"R2 score : {indicators['Indicators']['Indicators']['r2']}")


if(st.session_state['page']=='ViewModels'):
    st.title('Select model')
    data = display_model()
    print(data)
    df = pd.DataFrame(data=data["models"])
    st.write(df)
    st.write('Select a model :')
    print(np.array(df).shape)
    num_model = st.number_input("Enter the model's index of the model that you decide to chose :", 
                    min_value=0,
                    max_value=(np.array(df).shape)[1]-1)
    select = st.button('Select')
    if(select):
        data = requests.get(url=url_api+'/select/model', json={'index':num_model}).json()
        print(data)
        st.write(f"Mode {num_model} selected")


if(st.session_state['page']=='DeleteModel'):
    st.title('Delete model')
    data = display_model()
    df = pd.DataFrame(data=data["models"])
    st.write(df)
    st.write('Delete a model :')
    num_model = st.number_input("Enter the model's index of the model that you decide to delete :", 
                    min_value=0,
                    max_value=(np.array(df).shape)[1]-1)
    select = st.button('Delete')
    if(select):
        data = requests.delete(url=url_api+'/delete/model', json={'index':num_model}).json()
        print(data)
        st.write(f"Mode {num_model} deleted")

    
if(st.session_state['page']=='CurrentModel'):
    st.title('Current model')
    data = current_model()
    st.write(f"Your current model is : {data['Current-model']}")
    data = display_model()
    df = pd.DataFrame(data=data["models"])
    st.write(df)

if(st.session_state['page'] == 'CleanDataSet'):
    st.write('Clean your dataset')
    c = st.button('Clean it', key='clean_it_button')
    if(c):
        data = requests.get(url=url_api+'/clean-dataset').json()
        st.write(f"Clean is {data['Clean']}")

if(st.session_state['page'] == 'DataDrift'):
    st.title('Data Drift')
    st.write('Firstly, you have to run the data drift to update it and display it.')
    st.write('First step :')
    pdd = st.button('Process data drift')
    if(pdd):
        datadrif()
    st.write('Second step :')
    ddd = st.button('Display data drift')
    if(ddd):
        components.iframe(url_api + '/view/data-drift', height=1000, width=800, scrolling=True)


if(st.session_state['page'] == 'DataQuality'):
    st.title('Data Quality')
    st.write('Firstly, you have to run the data drift to update it and display it.')
    st.write('First step :')
    pdq = st.button('Process data quality')
    if(pdq):
        dataquality()
    st.write('Second step :')
    ddq = st.button('Display data quality')
    if(ddq):
        components.iframe(url_api + '/view/data-quality', height=1000, width=800, scrolling=True)
        

if(st.session_state['page'] == 'DataStability'):
    st.title('Data Stability')
    st.write('Firstly, you have to run the data drift to update it and display it.')
    st.write('First step :')
    pds = st.button('Process data stability')
    if(pds):
        datastability()
    st.write('Second step :')
    dds = st.button('Display data stability')
    if(dds):
        components.iframe(url_api + '/view/data-stability', height=1000, width=800, scrolling=True)

if(st.session_state['page'] == 'MSE'):
    st.title('Mean Squared Error')
    data = MSE()
    st.write(f"MSE : {data['RMSE']}")

if(st.session_state['page'] == 'MAE'):
    st.title('Mean Absolute Error')
    data = MAE()
    st.write(f"MAE : {data['MAE']}")

if(st.session_state['page'] == 'R2'):
    st.title('R2 score')
    data = R2()
    st.write(f"R2 : {data['R2']}")

if(st.session_state['page']=='ViewHistory'):
    st.title('Predictions history')
    components.iframe(url_api + '/view/prediction-history', height=400, scrolling=True)

if(st.session_state['page']=='AberantValues'):
    st.title('Aberant Values')
    components.iframe(url_api+'/view/aberant-values', height=150, scrolling=True)
    di = st.button('Delete it')
    if(di):
        data = requests.delete(url=url_api+'/delete/aberant-values').json()
        st.write(f"Delete {data['Delete']}")

if(st.session_state['page']=='CorrectValue'):
    st.title('Correct Y Value')
    data = requests.get(url=url_api+'/prediction-history').json()
    components.iframe(url_api+'/view/prediction-history', height=200, scrolling=True)
    st.write('Choose the row on which you want to replace Y and enter your value')
    df = pd.DataFrame(data=data)
    num_ligne = st.number_input("Enter the row's index :", 
                    min_value=0,
                    max_value=(np.array(df).shape)[0])
    value = st.text_input("Enter the value that will replace the current Y value :")
    num_ligne_button = st.button('Corrige')
    if(num_ligne_button):
        res = requests.put(url=url_api+'/correct-value', json={'value':value,'row':num_ligne}).json()
        st.write(f"Correct : {res['correct']}")