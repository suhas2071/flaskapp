from flask import Flask,request
import numpy as np
import pickle
with open('knn.pkl','rb') as model_file:
    model=pickle.load(model_file)

app=Flask(__name__)
@app.route('/predict')
def predict_iris():
    s_length = request.args.get("s_length")
    s_width = request.args.get("s_width")
    p_length = request.args.get("p_length")
    p_width = request.args.get("p_width")
    prediction=model.predict(np.array([[s_length,s_width,p_length, p_width]]))
    return str(prediction)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)

#http://127.0.0.1:7300/predict?s_length=6.0&s_width=3.4&p_length=4.5&p_width=1.6
