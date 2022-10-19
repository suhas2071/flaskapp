from flask import Flask,request
from flask import Flask,request
app=Flask(__name__)
@app.route('/')
def hello():
        return "Hello world"
@app.route('/suhas')
def suhas():
        return "Hello world Suhas"
app.run(host='0.0.0.0',port=8080)

