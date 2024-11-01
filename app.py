import numpy as np
import pandas as pd
import streamlit as st 
import sklearn
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
import xgboost as xgb

modelLR = joblib.load('model_LR.sav')
modelPoly = joblib.load('model_poly.sav')
modelRBF = joblib.load('model_rbf.sav')
modelAda = joblib.load('ada_clf2.sav')
modelBag = joblib.load('bag_clf.sav')
modelGradBoost = joblib.load('grad_boost_clf.sav')
modelDT = joblib.load('dt_best_smote_scaled.sav')
modelRF = joblib.load('rf_best_smote_scaled.sav')
modelVote = joblib.load('voting_clf.sav')
modelXGB = joblib.load('xgb_clf2.sav')

#pipeline = joblib.load('stroke_pipeline.sav')

def main():
    st.title("Stroke Prediction")
    gender = st.selectbox("gender",["Male","Female"])
    age = st.number_input("age",min_value=0.08,max_value=82.00,value="min")
    hypertension = st.selectbox("hypertension",[0,1])
    heart_disease = st.selectbox("heart disease",[0,1])
    ever_married = st.selectbox("ever married",["Yes","No"])
    work_type = st.selectbox("work type",["children","Govt_job","Never_worked","Private","Self-employed"])
    Residence_type = st.selectbox("Resindence Type",["Urban","Rural"])
    avg_glucose_level = st.number_input("avg glucose level",min_value=55.12,max_value=271.74,value="min")
    bmi = st.number_input("bmi",min_value=10.30,max_value=97.60,value="min")
    smoking_status = st.selectbox("Smoking Status",["formerly smoked","never smoked","smokes","Unknown"])
    select_model = st.selectbox("Select the model you want to try",["LogisticRegression","Poly-SVC","RBF-SVC","AdaBoost","Bagging","GradientBoost",
                                                                    "DecisionTree","RandomForest","Voting","XGBoost"])
    if st.button("Predict Stroke"):
        features = [[gender,age,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,
                     bmi,smoking_status]]
        data = {"gender":gender,"age":float(age),"hypertension":int(hypertension),
                "heart_disease":int(heart_disease),"ever_married":ever_married,"work_type":work_type,
                "Residence_type":Residence_type,"avg_glucose_level":float(avg_glucose_level),"bmi":float(bmi),
                "smoking_status":smoking_status}
        df = pd.DataFrame([list(data.values())],columns=["gender","age","hypertension","heart_disease","ever_married",
                                                         "work_type","Residence_type","avg_glucose_level","bmi","smoking_status"])
        labelEn = LabelEncoder()
        df["gender"] = labelEn.fit_transform(df["gender"])
        df["ever_married"] = labelEn.fit_transform(df["ever_married"])
        df["work_type"] = labelEn.fit_transform(df["work_type"])
        df["Residence_type"] = labelEn.fit_transform(df["Residence_type"])
        df["smoking_status"] = labelEn.fit_transform(df["smoking_status"])
        scaler = StandardScaler()
        df_prepared = scaler.fit_transform(df)
        #df_prepared = pipeline.fit_transform(df)
        if select_model == "LogisticRegression":
            prediction = modelLR.predict(df_prepared)
        elif select_model == "Poly-SVC":
            prediction = modelPoly.predict(df_prepared)
        elif select_model == "RBF-SVC":
            prediction = modelRBF.predict(df_prepared)
        elif select_model == "AdaBoost":
            prediction = modelAda.predict(df_prepared)
        elif select_model == "Bagging":
            prediction = modelBag.predict(df_prepared)
        elif select_model == "GradientBoost":
            prediction = modelGradBoost.predict(df_prepared)
        elif select_model == "DecisionTree":
            prediction = modelDT.predict(df_prepared)
        elif select_model == "RandomForest":
            prediction = modelRF.predict(df_prepared)
        elif select_model == "Voting":
            prediction = modelVote.predict(df_prepared)
        elif select_model == "XGBoost":
            prediction = modelXGB.predict(df_prepared)
        if prediction == 0:
            output = 'No Stroke'
        elif prediction == 1:
            output = 'Stroke'
        st.success('Prediction: {}'.format(output))

if __name__=='__main__': 
    main()