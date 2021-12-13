from typing import Tuple
from flask import Flask,render_template
import pickle
from flask.globals import request
import numpy as np
app = Flask(__name__)

def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 8)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
prediction="Kindly fill the above form!"
@app.route('/')
def hello_world():
    return render_template("home.html",prediction = prediction)

@app.route('/predict',methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = to_predict_list[0:8]
        print(to_predict_list)
        #to_predict_list = list(map(int, to_predict_list))
        to_predict_list[0]=int(to_predict_list[0])
        to_predict_list[1]=int(to_predict_list[1])
        to_predict_list[2]=int(to_predict_list[2])
        to_predict_list[3]=int(to_predict_list[3])
        to_predict_list[4]=int(to_predict_list[4])
        to_predict_list[5]=float(to_predict_list[5])
        to_predict_list[6]=float(to_predict_list[6])
        to_predict_list[7]=int(to_predict_list[7])
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)       
        if result== 0:
            prediction ="Not Diabetic"
            #print("not Diabetic")
        else:
            prediction ="Diabetic"
            #print("Diabetic")          
        return render_template("home.html", prediction = prediction)

if __name__ == "__main__":
    app.run()