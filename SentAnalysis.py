# Author： Rong Peng
# Date :  May08, 2020
# Description: P&G Test Code - Sentiment Analysis backend
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import numpy as np
from sklearn.feature_selection import RFE
import os
from warnings import simplefilter
# Filter the scikit-learn Future Warning
simplefilter(action='ignore', category=FutureWarning)

#Create the report.txt
reportpath="static/result/report.txt"
if os.path.exists(reportpath):
    os.remove(reportpath)
f=open(reportpath,"w+")

filepool=os.listdir('static/test')+os.listdir('static/train')
# print(filepool)

#Loading File
path_tain='static/train'

#Loading Training data
dataset_train = load_files(container_path=path_tain,categories=filepool)

# Loading Valuating data
test_path='static/test'
dataset_test=load_files(container_path=test_path,categories=filepool)
# print(dataset_train,dataset_test)


#Calculate the term frequency
# count_vect = CountVectorizer(decode_error='ignore')
# X_train_counts = count_vect.fit_transform(dataset_train.data)
# print(X_train_counts.shape)
# print(count_vect.get_feature_names())
# print(X_train_counts.toarray())

#Calculate TF-IDF
tf_transformer = TfidfVectorizer(decode_error='ignore')
X_train_counts_tf=tf_transformer.fit_transform(dataset_train.data)
# print(tf_transformer.get_feature_names())
# print(tf_transformer.vocabulary_)
voc=tf_transformer.vocabulary_
d=sorted(voc.items(), key=lambda item:item[1], reverse=True)
# print(d)
f.write("Top ten most frequently occurring words: \n")
j=0
for i in d:
    j = j + 1
    f.write("Word: "+i[0]+ " ,Frequency: ")
    f.write(str(i[1])+" \n")
    # print(i[0], i[1])
    if j==10:
        break
f.write("\n")

#Sentiment Analysis
Positive_count=0
Neg_count=0
Neutral_count=0
for i in voc:
    if i in ['like','love','favor','favourite','favorite','adore','cute','great','good','awesome','inspiring','wonderful','lovely','desired','desirable','famous','gorgeous','honor','honored','fortune','luck','lucky','fabulous']:
        Positive_count=Positive_count+1
    elif i in ['hate','damn','breakdown','sad','bad','shit','bored','annoyed','annoy','annoying','dumb','hateful','hard','gloomy','regret','sorry','Sorry','SORRY','ugly','infamous','unfortunate','unfortunately']:
        Neg_count=Neg_count+1
    else:
        Neutral_count=Neutral_count+1
f.write("Sentiment Analysis \n")
f.write("Positive: "+str(Positive_count)+" \n")
f.write("Negative: "+str(Neg_count)+" \n")
f.write("Netural: "+str(Neutral_count)+" \n")
f.write("\n")
#Matplotlib Figure
plt.figure(figsize=(3, 3), dpi=80)
N = 3
index = np.arange(3)
value=(Positive_count,Neutral_count,Neg_count)
width = 0.35
p2 = plt.bar(index, value, width, label="rainfall", color="#87CEFA")
plt.xlabel('Sentiment Classification')
plt.ylabel('CountNum')
plt.xticks(index, ('Positive', 'Netural', 'Negative'))
plt.yticks(np.arange(0,9500, 1000))
if os.path.exists('static/result/analysis.png'):
    os.remove('static/result/analysis.png')
plt.savefig('static/result/analysis.png',bbox_inches = 'tight')


# Baseline models setting
num_folds=10
seed=5
scoring='accuracy'

models={}
models['LR']=LogisticRegression()
models['SVM']=SVC()
models['CART']=DecisionTreeClassifier()
models['MNB']=MultinomialNB()
models['KNN']=KNeighborsClassifier()

results = []
MaxMean=0
Bestbaseline=''
f.write("Baseline models\n")
for key in models:
    temp=0
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_results = cross_val_score(models[key],X_train_counts_tf,dataset_train.target,cv=kfold,scoring=scoring)
    results.append(cv_results)
    temp=cv_results.mean()
    if temp > MaxMean:
      MaxMean=temp
      Bestbaseline=key
    f.write('%s: %f (%f)\n' % (key,cv_results.mean(),cv_results.std()))
    # print('%s: %f (%f)' % (key,cv_results.mean(),cv_results.std()))
# print(Bestbaseline,MaxMean)
f.write('Model '+ '%s'% Bestbaseline+ ' with the best accuracy '+ '%f'% MaxMean + '\n')

#Single model Tunning
# param_grid = {}
# param_grid['C']=[0.1 ,5,13,15]
# model = LogisticRegression()
# kfold = KFold(n_splits=num_folds,random_state=seed)
# grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
# grid_result=grid.fit(X=X_train_counts_tf,y=dataset_train.target)
# print('%s' % (grid_result.best_params_))
f.write("\n")


# #Ensemble
ensemble = {}
f.write("Ensemble models\n")
ensemble['RF']=RandomForestClassifier()
ensemble['AB']=AdaBoostClassifier()
results=[]
MaxMean2=0
Bestensemble=''
for key in ensemble:
    temp=0
    kfold = KFold(n_splits=num_folds,random_state=seed)
    cv_results = cross_val_score(ensemble[key],X_train_counts_tf,dataset_train.target,cv=kfold,scoring=scoring)
    results.append(cv_results)
    temp=cv_results.mean()
    if temp>MaxMean2:
        MaxMean2 = temp
        Bestensemble=key
    f.write('%s: %f(%f)\n'%(key, cv_results.mean(),cv_results.std()))
    # print('%s: %f(%f)'%(key, cv_results.mean(),cv_results.std()))
f.write("\n")
f.write('Ensemble Model '+ '%s'% Bestensemble+ ' with the best accuracy '+ '%f'% MaxMean2 + '\n')
if MaxMean>MaxMean2:
    finalModel=Bestbaseline
    # To judge if it is a baseline or ensemble
    flag='B'
else:
    finalModel=Bestensemble
    flag='E'

if flag=='E':
    M=ensemble[finalModel]
else:
    M=models[finalModel]
f.write('The appropriate classifier should be '+'%s'% M + "\n")


