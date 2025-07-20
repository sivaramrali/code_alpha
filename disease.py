import pandas as pd
df=pd.read_csv('hdp.csv')
print(df)

X=df.drop('TenYearCHD',axis=1)
Y=df['TenYearCHD']
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()

model.fit(X_train,Y_train)

predictions=model.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(predictions,Y_test)

import joblib
joblib.dump(model,'C:/Data/hdp.joblib')
clf=joblib.load('C:/Data/hdp.joblib')

import streamlit as st
import joblib
st.title('heart disease prediction')
model=joblib.load('C:/Data/hdp.joblib')
age=st.number_input('Enter your current age')
cigs_per_day=st.number_input('Enter how many cigarettes do you some every day')
tottal_chol=st.number_input('enter your total cholastral in your body')
sys_BP=st.number_input('enter your systolic blood pressure(BP) ')
dia_BP=st.number_input('enter your Diastolic blood pressure(BP) ')
BMI=st.number_input('enter you BMI ')
heart_rate=st.number_input('enter your heart rate')
glucose=st.number_input('enter your glucose levels')
if st.button('predict heart diease'):
    prediction=model.predict([[age,cigs_per_day,tottal_chol,sys_BP,dia_BP,BMI,heart_rate,glucose]])
    if prediction==1:
        st.markdown('<p style="color:red;">You may have chances to attack heart diease</p>',unsafe_allow_html=True)
    else:
        st.markdown('<p style="color:green;">you have no risk of heart diease</p>',unsafe_allow_html=True)
