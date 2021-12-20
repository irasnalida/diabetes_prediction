import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = joblib.load('model_scaler.pkl') 
#scaler = pickle.load(open(scalerfile, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    features = [np.array(int_features)]
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)

    output = prediction
    
    if output == 1:
        prediction='You Have Diabetes' 
    else:
        prediction='You Do Not Have Diabetes'
    return render_template('home.html', prediction=prediction, preg=int_features[0], gluc=int_features[1], bp=int_features[2], st=int_features[3], insu=int_features[4], bmi=int_features[5], dpf=int_features[6], age=int_features[7])

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run()