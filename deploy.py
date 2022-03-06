# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 00:06:19 2021

@author: abdal
"""
from flask_cors import CORS,cross_origin
import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load
from tensorflow.python.keras.models import load_model
app=Flask(__name__,template_folder='template')
model=load('diabetic_classfication.joblib')

app.config['SECRET_KEY'] = 'the quick brown fox jumps over the lazy   dog'
app.config['CORS_HEADERS'] = 'Content-Type'
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:port"}})



# Load your own trained model
@app.route('/')
def home():
    return render_template('front.html')

@app.route('/prediction',methods=['POST'])
def prediction():

   data1=float(request.form['a'])
   data2=float(request.form['b'])
   data3=float(request.form['c'])
   data4=float(request.form['d'])
   data5=float(request.form['e'])
   arr=np.array([[data1,data2,data3,data4,data5]])
   prediction=model.predict(arr)
   
   return str(prediction[0])

@app.route('/predict_api',methods=['POST','GET'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data1=float(request.form['data1'])
    data2=float(request.form['data2'])
    data3=float(request.form['data3'])
    data4=float(request.form['data4'])
    data5=float(request.form['data5'])
    arr=np.array([[data1,data2,data3,data4,data5]])
    prediction=model.predict(arr)
    output =str(prediction[0])
    return jsonify(output)



if __name__ == "__main__":
    app.run(host="localhost", port=5000)