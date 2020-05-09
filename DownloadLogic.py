# Author： Rong Peng
# Date :  May08, 2020
# Description: P&G Test Code - Download Page
import flask,os,sys
from flask import Flask, send_from_directory
interface_path = os.path.dirname(__file__)
sys.path.insert(0, interface_path)
app = flask.Flask(__name__, static_folder='static')
@app.route('/', methods=['GET'])
def index():
    return '<form action="/download" method="GET" enctype="multipart/form-data"> <button type="submit">Download result</button>' \
           '</br><p>Sentiment Analysis</p><br/><img src="static/result/analysis.png"/></form>'
@app.route("/download",methods=['GET'])
def download():
    filename='report.txt'
    return send_from_directory(r"static/result/", filename=filename, as_attachment=True)
print(app.url_map)
app.run(port=5000)
