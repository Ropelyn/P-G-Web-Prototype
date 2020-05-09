# Author： Rong Peng
# Date :  May08, 2020
# Description: P&G Test Code - Uploading Page
import flask, os, sys
from flask import request

interface_path = os.path.dirname(__file__)
sys.path.insert(0, interface_path)

server = flask.Flask(__name__, static_folder='static')

@server.route('/', methods=['get'])
def index():
    return '<form action="/upload" method="post" enctype="multipart/form-data">Category <input type="txt" id="category" name="category"><br/><br/>Training File Upload  <input type="file" id="uploadfile" name="trainfile"><br/><br/>' \
           'Testing File Upload  <input type="file" id="uploadfile2" name="testfile"><br/><br/><button type="submit">Upload</button></form>'

@server.route('/upload', methods=['POST'])
def upload():
    fname = request.files['trainfile']
    f2name = request.files['testfile']
    category=request.form.get('category')
    print(category)
    dir_train=r'static/train/'+ category
    dir_test = r'static/test/' + category
    print(dir_train)
    if not os.path.exists(dir_train):
        os.mkdir(dir_train,mode=0o777)
    if not os.path.exists(dir_test):
        os.mkdir(dir_test, mode=0o777)
    if category and fname and f2name:
        train_fname =  r'static/train/' + category+'/'+ fname.filename
        fname.save(train_fname)
        test_fname = r'static/test/' + category+'/'+f2name.filename
        f2name.save(test_fname)
        return 'Done the Uploading'
    else:
        return '{"msg": "Please upload your testing and trainning files！"}'

print('----------Route Relationship----------')
print(server.url_map)
server.run(port=8000)