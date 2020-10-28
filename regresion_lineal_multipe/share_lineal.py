import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')
from sklearn.model_selection import train_test_split
from sklearn import metrics
from joblib import load
import streamlit as st
#--------------------------------------------#
#titulo#
#--------------------------------------------#
st.title('Regresion Lineal Multiple con streamlit')

#--------------------------------------------#
#Cargar los datos#
#--------------------------------------------#
df = pd.read_csv('Documents/Cursos de ML/aprende machine learning/articulos_ml.csv')
df = df.dropna()
st.write(df.head())

#--------------------------------------------#
#dimension los datos#
#--------------------------------------------#
'**cantidad de filas=**', df.shape[0], '**cantidad de colunas =**', df.shape[1]

#--------------------------------------------#
#sidebar-slider#
#--------------------------------------------#
st.sidebar.header('Parametros especifícos')
#['Word count', '# of Links', '# Images video', 'Elapsed days']

def input_features():
    Word_count = st.sidebar.slider('Word count', df['Word count'].min(), df['Word count'].max())
    No_of_links = st.sidebar.slider('No_of_link', df['# of Links'].min(), df['# of Links'].max())
    No_images_video= st.sidebar.slider('No. Images Videot', df['# Images video'].min(), df['# Images video'].max())
    Elapsed_days= st.sidebar.slider('Elapsed days', df['Elapsed days'].min(), df['Elapsed days'].max())
    
    data = {
        'Word_count': Word_count, 
    'No_of_links': No_of_links,
           'No_images_video' : No_images_video,
            'Elapsed_days': Elapsed_days 
        }
    features = pd.DataFrame(data, index=[0])
    return features

st.header('Variables seleccionadas')
df2 = input_features()
st.table(df2)

#--------------------------------------------#
#Cargar modelor#
#--------------------------------------------#
modelo = load('Documents/Cursos de ML/aprende machine learning/Regresion_lineal_multiple.joblib')

#--------------------------------------------#
#Visualizar y predecir#
#--------------------------------------------#
st.write('**Predicción de las veces a compartir**')
prediccion = modelo.predict(df2)
st.write(prediccion)
st.write('---')

