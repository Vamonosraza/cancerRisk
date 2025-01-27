import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# load the model to get the scores of the features
model = joblib.load('cervical_cancer_model.pkl')

st.title('Cervical Cancer Information')

# questions to ask the user
age = st.number_input('Age', min_value=0, max_value=100, value=20)
smokes = st.selectbox('Do you smoke?', ['Yes', 'No'])
hormonal_contraceptives = st.selectbox('Do you use hormonal contraceptives?', ['Yes', 'No'])
had_stds = st.selectbox('Have you ever had STDs?', ['Yes', 'No'])
dx = st.selectbox('Have you been diagnosed with any conditions?', ['Yes', 'No'])
dx_hpv = st.selectbox('Have you been diagnosed with HPV?', ['Yes', 'No'])

# convert the user inputs to the expected format
if st.button('Information'):
    info = []

    if age > 30:
        info.append("Age over 30 is a risk factor for cervical cancer. Regular screening is recommended.")
    if smokes == 'Yes':
        info.append("Smoking is a risk factor for cervical cancer. It can cause changes in the cervical cells that may lead to cancer.")
    if hormonal_contraceptives == 'Yes':
        info.append("Long-term use of hormonal contraceptives can increase the risk of cervical cancer. It's important to discuss the risks and benefits with your healthcare provider.")
    if had_stds == 'Yes':
        info.append("Having a history of STDs can increase the risk of cervical cancer. STDs can cause chronic inflammation and changes in the cervical cells.")
    if dx == 'Yes':
        info.append("Being diagnosed with certain conditions, such as cervical dysplasia or other precancerous conditions, can increase the risk of cervical cancer.")
    if dx_hpv == 'Yes':
        info.append("HPV infection is a major risk factor for cervical cancer. Regular screening and vaccination can help reduce the risk.")

    if not info:
        st.write("Based on your inputs, there are no significant risk factors for cervical cancer. However, regular screening is still important.")
    else:
        for message in info:
            st.write(message)
        st.write("Although you may have several risk factors, the chances of getting cervical cancer are still low. Regular check-ups and a healthy lifestyle can help manage these risks.")

# graph the top features according to the model
if st.button('Show Feature Importance'):
    st.write(
        "The Feature Importance graph shows the relative importance of each feature used by the model to make predictions. Features with higher importance values have a greater impact on the model's predictions.")
    feature_names = [
        'Age', 'Number of sexual partners', 'First sexual intercourse', 'Num of pregnancies', 'Smokes',
        'Smokes (years)', 'Smokes (packs/year)', 'Hormonal Contraceptives', 'Hormonal Contraceptives (years)',
        'IUD', 'IUD (years)', 'STDs', 'STDs (number)', 'STDs:condylomatosis', 'STDs:cervical condylomatosis',
        'STDs:vaginal condylomatosis', 'STDs:vulvo-perineal condylomatosis', 'STDs:syphilis',
        'STDs:pelvic inflammatory disease', 'STDs:genital herpes', 'STDs:molluscum contagiosum', 'STDs:AIDS',
        'STDs:HIV', 'STDs:Hepatitis B', 'STDs:HPV', 'STDs: Number of diagnosis', 'Dx:Cancer', 'Dx:CIN',
        'Dx:HPV', 'Dx', 'Hinselmann', 'Schiller', 'Citology'
    ]

    booster = model.get_booster()
    importance = booster.get_score(importance_type='weight')
    importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    keys = [feature_names[int(k[1:])] for k, v in importance]
    values = [v for k, v in importance]

    plt.figure(figsize=(10, 5))
    plt.bar(keys, values)
    plt.title('Feature Importance')
    plt.xticks(rotation=90)
    plt.show()
    st.pyplot(plt)

# show the correlation matrix heatmap
if st.button('Show Correlation Matrix Heatmap'):
    st.write(
        "The Correlation Matrix Heatmap shows the correlation coefficients between different features in the dataset. A higher absolute value indicates a stronger correlation, with positive values indicating a positive correlation and negative values indicating a negative correlation.")
    file_path = 'cervical_cancer.csv'
    if os.path.exists(file_path):
        cancer_df = pd.read_csv(file_path)  # Load your dataset
        cancer_df.replace('?', np.nan, inplace=True)  # Replace '?' with NaN
        cancer_df.dropna(inplace=True)  # Drop rows with NaN values
        cancer_df = cancer_df.drop(columns=['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])
        cancer_df = cancer_df.apply(pd.to_numeric, errors='coerce')
        cancer_df.fillna(cancer_df.mean(), inplace=True)
        corr_matrix = cancer_df.corr()

        plt.figure(figsize=(30, 30))
        sns.heatmap(corr_matrix, annot=True)
        plt.title('Correlation Matrix Heatmap')
        st.pyplot(plt)
    else:
        st.error(f"File not found: {file_path}")

# acknowledgment and disclaimer
st.write("### Acknowledgment")
st.write("This project was partially guided by Ryan Ahmed through Coursera.")
st.write("### Disclaimer")
st.write("This application is not intended to provide medical advice. It is a demonstration of data analysis and machine learning techniques applied to datasets.")