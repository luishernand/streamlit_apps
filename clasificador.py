##############
# Libraries
##############
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

##############################
#titulos, encabezados
##############################
st.title('Streamllit example')
st.write('''
# Explore different classifier
    ''')

##############################
#slider
##############################
datasets_names = st.sidebar.selectbox('Select Dataset', ('Iris', 'Breast Cancer', 'Wine dataset'))
classifier_name = st.sidebar.selectbox('Select classifier', ('KNN', 'SVM', 'Random Forest'))
st.write(f'## {datasets_names}')
################################
# Load datastes
################################
def get_dataset(datasets_names):
    if datasets_names == 'Iris':
        data = datasets.load_iris()
    elif datasets_names == 'Breast Cancer':
        data = datasets.load_breast_cancer()
    else:
        data =  datasets.load_wine()
    X = data.data
    y = data.target
    z = data.target_names
    j = data.feature_names
    return X, y,z,j

X, y, z,j = get_dataset(datasets_names)

df = pd.DataFrame(X, columns = j)
df['target'] = y

##################################
# General Info
##################################
st.write(df)
st.write('Target Names:', z)
st.write('shape of dataset', X.shape)
clas = len(np.unique(y))
st.write('Number of class', clas)
balance = df.target.value_counts(normalize=True)
st.write('% of class', balance)
###################################
# Statistics
###################################
st.subheader('Statistics')
st.write(df.describe())


###################################
# input parameters
###################################
def input_parameters(clf_name):
    params = dict()
    if clf_name == 'KNN':
        K = st.sidebar.slider("K", 1,15)
        params['K'] = K
    elif clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01,10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('max_depth', 2,15)
        n_estimator = st.sidebar.slider('n_estimator', 1,100)
        params['max_depth'] = max_depth
        params['n_estimator'] = n_estimator
    return params



params = input_parameters(classifier_name)


######################################
#classifier
######################################
def get_classifier(clf_name, params):
    clf = None
    if clf_name == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=params['K'])
    elif clf_name == 'SVM':
        clf = SVC(C = params['C'])
    else:
        clf = RandomForestClassifier(n_estimators = params['n_estimator'],
        max_depth = params['max_depth'], random_state = 1234)
    return clf

clf = get_classifier(classifier_name, params)

##############
# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 1234)
###############

#############
# Training and predicted
############
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

##############
#accuracy
#############
acc = accuracy_score(y_test, y_pred)

########################
# display predicted and accuracy
############################
st.write(f'classifier = {classifier_name}')
st.write(f'accuracy = {acc}')


##############################
# plot
#############################

pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[: ,0]
x2 = X_projected[: ,1]


fig = plt.figure()
plt.scatter(x1, x2, c= y, alpha = 0.8, cmap = 'viridis')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.colorbar()
st.pyplot()
