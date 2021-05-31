import pickle
import numpy as np
import pandas as pd

path1='C:\\Git Repos\\Lab-Report-Interpretor\\Models\\prediction-model-diabetes.pkl'
path2='C:\\Git Repos\\Lab-Report-Interpretor\\Models\\prediction-model-heart-attack.pkl'
path3='C:\\Git Repos\\Lab-Report-Interpretor\\Models\\prediction-model-liver-disease.pkl'

classifier = pickle.load(open(path1, 'rb'))
classifier1 = pickle.load(open(path2, 'rb'))
classifier2 = pickle.load(open(path3, 'rb'))


from flask import Flask, render_template, request
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Diabetes')
def Diabetes():
    return render_template('Diabetes.html')

@app.route('/CardiacArrest')
def CardiacArrest():
    return render_template('CardiacArrest.html')

@app.route('/LiverDisease')
def LiverDisease():
    return render_template('LiverDisease.html')

@app.route('/D4')
def D4():
    return render_template('D4.html')

@app.route('/D5')
def D5():
    return render_template('D5.html')

@app.route('/Predict', methods=['POST'])
def Predict():
    if request.method == 'POST':
        A = float(request.form['A'])
        B = float(request.form['B'])
        C = float(request.form['C'])
        D = float(request.form['D'])
        E = float(request.form['E'])
        F = float(request.form['F'])
        G = float(request.form['G'])
        H = float(request.form['H'])
        
        my_prediction = classifier.predict([[A,B,C,D,E,F,G,H]])

        return render_template('result.html', prediction=str(my_prediction[0]))

@app.route('/Predict1', methods=['POST'])
def Predict1():
    if request.method == 'POST':
        A = float(request.form['A'])
        B = float(request.form['B'])
        C = float(request.form['C'])
        D = float(request.form['D'])
        E = float(request.form['E'])
        F = float(request.form['F'])
        G = float(request.form['G'])
        H = float(request.form['H'])
        I = float(request.form['I'])
        J = float(request.form['J'])
        K = float(request.form['K'])
        L = float(request.form['L'])
        M = float(request.form['M'])
        data={'age':A, 'sex':B, 'cp':C, 'trtbps':D, 'chol':E, 'fbs':F, 'restecg':G, 'thalachh':H, 'exng':I, 'oldpeak':J, 'slp':K, 'caa':L, 'thall':M}
        df = pd.DataFrame(data,index=[0])
        my_prediction = classifier1.predict(df)

        return render_template('result.html', prediction=str(my_prediction[0]))

@app.route('/Predict2', methods=['POST'])
def Predict2():
    if request.method == 'POST':
        A = float(request.form['A'])
        B = float(request.form['B'])
        C = float(request.form['C'])
        D = float(request.form['D'])
        E = float(request.form['E'])
        F = float(request.form['F'])
        
        my_prediction = classifier2.predict([[A,B,C,D,E,F]])
        if my_prediction[0]==1:
            return render_template('result.html', prediction='0')
        else:
            return render_template('result.html', prediction='1')


        

if __name__ == '__main__':
    app.run(debug=True)
