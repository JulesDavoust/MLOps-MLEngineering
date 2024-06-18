from flask import Flask, request, jsonify
import joblib
from routes import model
from routes import intern

app = Flask(__name__)

intern.intern_routes(app)
#/
model.prevision_routes(app)
# #/prevision/predict
# #/prevision/batch-predict
# history.history_routes(app)
# #/predictions-history
# status.status_routes(app)
# #/status
# model.model_routes(app)
# #/model-info
# retrain.retrain_routes(app)
# #/retrain
# stat_perf.stat_perf_routes(app)
# #/stats-performance

# load_model = joblib.load('./models/model.pkl')

# data = [[245, 11, 9]]

# y_pred = load_model.predict(data)
# print(y_pred)


if __name__ == "__main__":
    app.run(debug=True)