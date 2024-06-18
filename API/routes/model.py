from flask import request, jsonify, render_template_string
import numpy as np
import pandas as pd
from functions.model import json_aberant_values, json_models, indice_model, current_model, select_models, delete_model, view_models, correct_value_y, del_aberant_values, predict, batch_predict, clean_dataset, run_data_drift_test_suite, run_data_quality_test_suite, run_data_stability_test_suite, result_data_drift_test_suite, result_data_quality_test_suite, result_data_stability_test_suite, MSE, MAE, R2, history, retrain, version, info_model_coef, info_model_intercept, give_data, aberant_values

def prevision_routes(app):
    @app.route('/prevision/predict', methods=['POST'])
    def post_predict():
        data = np.array((request.get_json())['features'])
        if data.ndim == 1:
            data = np.array([data])
        y_pred = predict(data)
        print("y_pred : ",y_pred)
        return jsonify({"y_pred":y_pred.tolist()})
    
    @app.route('/prevision/batch-predict', methods=['POST'])
    def post_batch_predict():
        data = np.array((request.get_json())['features'])
        if data.ndim == 1:
            data = np.array([data])
        y_pred = batch_predict(np.array(data))
        return jsonify({"y_pred":y_pred.tolist()})
    
    #DATA DRIFT
    @app.route('/monitoring/process/data-drift', methods=['POST'])
    def post_data_drift():
        reference_data = pd.read_csv('advertising_clean.csv')
        current_data = pd.read_csv('new_advertising.csv')
        res = run_data_drift_test_suite(reference=reference_data, current=current_data)
        if res:
            return jsonify({"message": "Monitoring data drift initiated"}), 202
        else:
            return jsonify({res})
        
    @app.route('/monitoring/result/data-drift', methods=['GET'])
    def get_data_drift():
        res = result_data_drift_test_suite()
        return res, 200, {'Content-Type': 'text/html'}
        
    @app.route('/view/data-drift')
    def view_data_drift():
        res = result_data_drift_test_suite()
        return render_template_string(res)
    
    #DATA QUALITY
    @app.route('/monitoring/process/data-quality', methods=['POST'])
    def post_data_quality():
        reference_data = pd.read_csv('advertising_clean.csv')
        current_data = pd.read_csv('new_advertising.csv')
        res = run_data_quality_test_suite(reference=reference_data, current=current_data)
        if res:
            return jsonify({"message": "Monitoring data quality initiated"}), 202
        else:
            return jsonify({res})

    @app.route('/monitoring/result/data-quality', methods=['GET'])
    def get_data_quality():
        res = result_data_quality_test_suite()
        return res, 200, {'Content-Type': 'text/html'}
        
    @app.route('/view/data-quality')
    def view_data_quality():
        res = result_data_quality_test_suite()
        return render_template_string(res)
    
    #DATA STABILITY
    @app.route('/monitoring/process/data-stability', methods=['POST'])
    def post_data_stability():
        reference_data = pd.read_csv('advertising_clean.csv')
        current_data = pd.read_csv('new_advertising.csv')
        res = run_data_stability_test_suite(reference=reference_data, current=current_data)
        if res:
            return jsonify({"message": "Monitoring data stability initiated"}), 202
        else:
            return jsonify({res})
        
    @app.route('/monitoring/result/data-stability', methods=['GET'])
    def get_data_stability():
        res = result_data_stability_test_suite()
        return res, 200, {'Content-Type': 'text/html'}
        
    @app.route('/view/data-stability')
    def view_data_stability():
        res = result_data_stability_test_suite()
        return render_template_string(res)
    
    @app.route('/monitoring/MSE', methods=['GET'])
    def get_MSE():
        mse = MSE()
        return jsonify({'RMSE': mse})
    
    @app.route('/monitoring/MAE', methods=['GET'])
    def get_MAE():
        mae = MAE()
        return jsonify({'MAE': mae})
    
    @app.route('/monitoring/R2', methods=['GET'])
    def get_R2():
        r2 = R2()
        print(r2)
        return jsonify({"R2": r2})
    
    @app.route('/clean-dataset', methods=['GET'])
    def get_clean_dataset():
        clean_dataset()
        return jsonify({'Clean':'ok'})
    
    @app.route('/prediction-history', methods=['GET'])
    def get_prediction_history():
        ph = history()
        return jsonify({'History ':ph.to_dict()})
    
    @app.route('/view/prediction-history')
    def view_prediction_history():
        ph = history()
        df = ph.to_html(classes="table table-striped")
        return render_template_string(df)
    
    @app.route('/retrain-model', methods=['GET'])
    def get_retrain():
        df = retrain()
        return jsonify({'verif':df.to_dict()})
    
    @app.route('/version-model', methods=['GET'])
    def get_version():
        v = version()
        return jsonify({'version':v})
    
    @app.route('/informations-model/coef', methods=['GET'])
    def get_coef():
        coef = info_model_coef()
        print(coef)
        return jsonify({'coeffecients':coef.tolist()})

    @app.route('/informations-model/intercept', methods=['GET'])
    def get_intercept():
        intercept = info_model_intercept()
        return jsonify({'intercept':intercept})

    @app.route('/give-data', methods=['POST'])
    def get_give_data():
        data = np.array((request.get_json())['data'])
        if data.ndim == 1:
            data = np.array([data])
            print(data)
        resp = give_data(np.array(data))
        print(resp)
        return jsonify({'data-gived':resp})
    
    @app.route('/view/aberant-values')
    def view_aberant_values():
        df = aberant_values()
        df = df.to_html(classes="table table-striped")
        return render_template_string(df)
    
    @app.route('/json/aberant-values')
    def get_aberant_values():
        df = json_aberant_values()
        return df
    
    @app.route('/delete/aberant-values', methods=['DELETE'])
    def delete_aberant_values():
        del_aberant_values()
        return jsonify({'Delete':'did'})
    
    @app.route('/correct-value', methods=['PUT'])
    def put_correct_value():
        value = request.get_json()['value']
        row = request.get_json()['row']
        correct_value_y(value=value, row=row)
        return jsonify({'correct':'did'})
    
    @app.route('/view/models', methods=['GET'])
    def view_models_():
        df = view_models()
        df = df.to_html(classes='table table-striped')
        return render_template_string(df)
    
    @app.route('/json/models', methods=['GET'])
    def get_models():
        df = json_models()
        return jsonify({'models':df.to_dict()})
    
    @app.route('/delete/model', methods=['DELETE'])
    def del_model():
        index = request.get_json()['index']
        res = delete_model(index)
        return jsonify({'Delete':res})
    
    @app.route('/select/model', methods=['GET'])
    def sel_model():
        index = request.get_json()['index']
        res = select_models(index)
        return jsonify({'Select':'did'})
    
    @app.route('/current-model', methods=['GET'])
    def get_model():
        res = current_model()
        return jsonify({'Current-model':res})
    
    @app.route('/indicator-model', methods=['GET'])
    def get_indicator_model():
        index = request.get_json()['index']
        df = indice_model(index)
        return jsonify({'Indicators':df.to_dict()})
    
    