import numpy as np
from flask import Flask, render_template, request
import pickle

#Create flask app
app = Flask(__name__)
model = pickle.load(open("heartdisease.pkl", "rb"))
classifier = pickle.load(open("diabetes.pkl", "rb"))
model1 = pickle.load(open("parkinson.pkl", "rb"))
model2 = pickle.load(open("liver.pkl", "rb"))

@app.route("/")
def home():
    result = ''
    return render_template("home2.html", **locals())

@app.route("/predict", methods = ["POST", "GET"])
def predict():
    age = float(request.form['age'])
    sex = float(request.form['sex'])
    cp = float(request.form['cp'])
    trestbps = float(request.form['trestbps'])
    chol = float(request.form['chol'])
    fbs = float(request.form['fbs'])
    restecg = float(request.form['restecg'])
    thalach = float(request.form['thalach'])
    exang = float(request.form['exang'])
    oldpeak = float(request.form['oldpeak'])
    slope = float(request.form['slope'])
    ca = float(request.form['ca'])
    thal = float(request.form['thal'])
    result = model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])[0]
    if result == 0:
        r1 = 'The person does hot have a Heart Disease'
    else:
        r1 = 'The person has Heart Disease'

    return render_template("home2.html", **locals())

@app.route("/predict1", methods = ["POST", "GET"])
def predict1():
    Pregnancies = float(request.form['Pregnancies'])
    Glucose = float(request.form['Glucose'])
    BloodPressure = float(request.form['BloodPressure'])
    SkinThickness = float(request.form['SkinThickness'])
    Insulin = float(request.form['Insulin'])
    BMI = float(request.form['BMI'])
    DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
    Age = float(request.form['Age'])
    result = classifier.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])[0]
    if result == 0:
        r2 = 'The person is not diabetic'
    else:
        r2 = 'The person is diabetic'

    return render_template("home2.html", **locals())

@app.route("/predict2", methods = ["POST", "GET"])
def predict2():
    Fo = float(request.form['in1'])
    Fhi = float(request.form['in2'])
    Flo = float(request.form['in3'])
    Jitter = float(request.form['in4'])
    JitterAbs = float(request.form['in5'])
    RAP = float(request.form['in6'])
    PPQ = float(request.form['in7'])
    DDP = float(request.form['in8'])
    Shimmer = float(request.form['in9'])
    ShimmerdB = float(request.form['in10'])
    APQ3 = float(request.form['in11'])
    APQ5 = float(request.form['in12'])
    APQ = float(request.form['in13'])
    DDA = float(request.form['in14'])
    NHR = float(request.form['in15'])
    HNR = float(request.form['in16'])
    RPDE = float(request.form['in17'])
    DFA = float(request.form['in18'])
    spread1 = float(request.form['in19'])
    spread2 = float(request.form['in20'])
    D2 = float(request.form['in21'])
    PPE = float(request.form['in22'])
    result = model1.predict([[Fo, Fhi, Flo, Jitter, JitterAbs, RAP, PPQ, DDP, Shimmer, ShimmerdB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])[0]

    if result == 0:
        r3 = 'The Person does not have Parkinsons Disease'
    else:
        r3 = 'The Person has Parkinsons Disease'

    return render_template("home2.html", **locals())

@app.route("/predict3", methods = ["POST", "GET"])
def predict3():
    Age = float(request.form['Age'])
    Gender = float(request.form['Gender'])
    Total_Bilirubin = float(request.form['Total_Bilirubin'])
    Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
    Alkaline_Phosphotase = float(request.form['Alkaline_Phosphotase'])
    Alamine_Aminotransferase = float(request.form['Alamine_Aminotransferase'])
    Aspartate_Aminotransferase = float(request.form['Aspartate_Aminotransferase'])
    Total_Protiens = float(request.form['Total_Protiens'])
    Albumin = float(request.form['Albumin'])
    Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
    result = model2.predict([[Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio]])[0]

    if result == 0:
        r4 = 'The person does hot have a Liver Disease'
    else:
        r4 = 'The person has Liver Disease'

    return render_template("home2.html", **locals())



if __name__ == "__main__":
    app.run(debug=True)