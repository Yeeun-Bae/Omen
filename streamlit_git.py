import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import ast
import seaborn as sns
import matplotlib.pyplot as plt


placeholder=st.empty()
if "page" not in st.session_state:
    st.session_state.page = 0
if "question" not in st.session_state:
    st.session_state.question = 0

if "personal" not in st.session_state:
    st.session_state.personal = pd.DataFrame(columns =  ['symptom_desc', 'class_desc'])

if "patient_data" not in st.session_state:
    st.session_state.patient_data = pd.DataFrame(columns =  ['disease', 'input']) 

def diagnose(): 
    st.session_state.diseases = set(symptoms[symptoms['chosen']]['diseases'].sum())
    st.session_state.page = 1

def reset():
    st.session_state.page = 0
    st.session_state.question = 0
    st.session_state.personal = pd.DataFrame(columns =  ['symptom_desc', 'class_desc'])
    st.session_state.patient_data = pd.DataFrame(columns =  ['disease', 'input']) 

def results():
    st.session_state.page = 2

def next_q(answer, q):
    st.session_state.question +=1
    new  = pd.DataFrame([[q, answer]], columns = ['symptom_desc', 'class_desc'])
    st.session_state.personal = pd.concat([st.session_state.personal, new], axis = 'index', ignore_index=True)

def mk_model_in(dis, df):
    if ~(dis in st.session_state.patient_data['disease']):
        new = pd.DataFrame([[dis, dict(zip(df['symptom'], df['value']))]], columns = ['disease', 'input'])
        st.session_state.patient_data = pd.concat([st.session_state.patient_data, new], axis = 'index', ignore_index=True)
    

symptoms = pd.DataFrame(data=np.array([
    [False, 'Education', 'Personal Background', ['Alzheimer', 'Stroke']],
    [False,'Wealth', 'Personal Background', ['Alzheimer', 'Stroke']],
    [False,'Smoking', 'Personal Background', ['Lung_Cancer','Stroke']],
    [False,'Alcohol', 'Personal Background', ['Lung_Cancer']],
    [False,'Anxiety', 'Mental/Cognitive', ['Lung_Cancer']],
    [False,'Appetite Loss', 'Mental/Cognitive', ['Kidney_Disease']],
    [False,'Peer Pressure', 'Mental/Cognitive', ['Lung_Cancer']],
    [False,'Forgetfulness', 'Mental/Cognitive', ['Alzheimer']],
    [False,'Blood Pressure', 'Cardiovascular', ['Heart_Disease', 'Kidney_Disease', 'Stroke']],
    [False,'Heart Rate', 'Cardiovascular', ['Heart_Disease']],
    [False,'Blood Sugar', 'Cardiovascular', ['Heart_Disease', 'Stroke', 'Kidney_Disease']],
    [False,'ECG Abnormality', 'Cardiovascular', ['Heart_Disease']],
    [False,'Chest Pain', 'Cardiovascular', ['Heart_Disease']],
    [False,'Abnormal Lab Values', 'Cardiovascular', ['Heart_Disease', 'Kidney_Disease']],
    [False,'Wheezing', 'Respiratory', ['Lung_Cancer']],
    [False,'Allergies', 'Respiratory', ['Lung_Cancer']],
    [False,'Coughing', 'Respiratory', ['Lung_Cancer']],
    [False,'Shortness of Breath', 'Respiratory', ['Lung_Cancer']],
    [False,'Trouble Swallowing', 'Respiratory', ['Lung_Cancer']],
    [False,'Diabetes', 'Pre-existing Conditions',['Kidney_Disease', 'LungCancer']],
    [False,'Coronary Artery Disease', 'Pre-existing Conditions', ['Kidney_Disease', 'Lung_Cancer']],
    [False,'Heart Disease', 'Pre-existing Conditions', ['Lung_Cancer', 'Stroke']],
    [False,'Other Chronic Disease', 'Pre-existing Conditions', ['Lung_Cancer']]
    ], dtype = object), columns = ['chosen', 'name', 'group', 'diseases'])


if st.session_state.page == 0:
    with placeholder.container():
        for group in symptoms["group"].unique():   
            with st.expander(group):
                for symp in symptoms[symptoms["group"]==group]["name"]:
                    symptoms.loc[symptoms["name"] == symp, 'chosen'] = st.checkbox(symp, False)
        st.button('Diagnosis',on_click=diagnose, type='primary', disabled = ~symptoms['chosen'].any())

elif st.session_state.page == 1:
    with placeholder.container():
        ds = pd.read_csv(str(Path.cwd())+r'/disease_symptoms.csv')

        for idx, q in enumerate(ds[ds['disease'].isin(st.session_state.diseases)]['symptom_desc'].unique()):
            if st.session_state.question == idx:
                st.write(f"Please provide additional information on patient's {q}:")
                symps = ds[ds['symptom_desc']==q]
                is_cont = ~symps['class'].str.contains(',').all()
                
                if(is_cont):
                    thresholds = list(map(float,(symps[(~ds['class'].str.contains(','))]["class"].str.split(' - ').values.sum())))
                    maximum = max(thresholds)
                    minimum = min(thresholds)
                    label = f"{minimum} - {maximum}"
                    answer = st.text_input(label = label, key = q)
                            
                elif(~is_cont):
                    translation = ~symps['class_desc'].isna().all()

                    if(translation):
                        options = list(ast.literal_eval(symps[~symps['class_desc'].isna()]['class_desc'].iloc[0]).keys())
                    else:
                        here = symps['class'].str.len().nlargest(1).index
                        options = ast.literal_eval(ds['class'].iloc[here].values[0])

                    answer = st.selectbox(label='', options = options, key = q, label_visibility = 'collapsed')
                             
                st.button("Next", on_click=next_q, args=[answer, q])               
        
        if st.session_state.question == len(ds[ds['disease'].isin(st.session_state.diseases)]['symptom_desc'].unique()):
            st.write(st.session_state.personal)
            for dis in st.session_state.diseases:
                this = st.session_state.personal.merge(ds[ds['disease']==dis], on='symptom_desc', how='inner')
                this.loc[this['class_desc_y'].isna(),'value']= this['class_desc_x']
                for ii in this[this['bnr'].isna() & ~this['class_desc_y'].isna()].index:
                    this.iloc[ii, this.columns.get_indexer(['value'])] = eval(this.iloc[ii]['class_desc_y'])[this.iloc[ii]['class_desc_x']]
                for ii in this[~this['bnr'].isna()].index:
                    this.iloc[ii, this.columns.get_indexer(['value'])] = eval(this.iloc[ii]['class_desc_y'])[float(this.iloc[ii]['class_desc_x']) > float(this.iloc[ii]['bnr'])]

                #st.write( dict(zip(this['symptom'], this['value'])))    
                mk_model_in(dis, this)
                #st.text(st.session_state.patient_data)
            
            st.button('Show Results',on_click=results, type='primary')   
        #st.write(answer)       
        
        

elif st.session_state.page == 2:
    with placeholder.container():
        diseases = []
        probabilities = []
        for idx, disease in enumerate(st.session_state.patient_data["disease"]):
            input = pd.DataFrame(st.session_state.patient_data["input"][idx], index=[0,])
            if disease == 'Alzheimer':
                with open(str(Path.cwd())+r'/alzheimer_logistic.pkl', 'rb') as a:
                    alzheimer_logistic = pickle.load(a)
                    diseases.append(disease)
                    probabilities.append(round(alzheimer_logistic.predict_proba(input)[0][1], 2) * 100)
            elif disease == 'Stroke':
                with open(str(Path.cwd())+r'/stroke_logistic.pkl', 'rb') as s:
                    stroke_logistic = pickle.load(s)
                    diseases.append(disease)
                    probabilities.append(round(stroke_logistic.predict_proba(input)[0][1], 2) * 100)
            elif disease == 'Heart_Disease':
                with open(str(Path.cwd())+r'/heart_logistic.pkl', 'rb') as h:
                    heart_logistic = pickle.load(h)
                    diseases.append(disease)
                    probabilities.append(round(heart_logistic.predict_proba(input)[0][1], 2) * 100)
            elif disease == 'Kidney_Disease':
               with open(str(Path.cwd())+r'/kidney_logistic.pkl', 'rb') as k:
                   kidney_logistic = pickle.load(k)
                   diseases.append(disease)
                   probabilities.append(round(kidney_logistic.predict_proba(input)[0][1], 2) * 100)
            elif disease == 'Lung_Cancer':
               with open(str(Path.cwd())+r'/lungcancer_logistic.pkl', 'rb') as l:
                   lungcancer_logistic = pickle.load(l)
                   diseases.append(disease)
                   probabilities.append(round(lungcancer_logistic.predict_proba(input)[0][1], 2) * 100)

        prob_df = pd.DataFrame({'Disease':diseases,'Likelihood(%)':probabilities})
        #st.write(prob_df)
        
        sns.set_style('darkgrid')
        sns.set_palette('Set2')
        ax = sns.barplot(data=prob_df, x="Disease", y="Likelihood(%)")
        ax.bar_label(ax.containers[0])
        ax.set_yticks([0, 25, 50, 75, 100])

        #plt.title('Age and Class of Titanic Passengers')
        plt.ylabel('Likelihood(%)')
        #sns.despine()
        st.pyplot(ax.get_figure())

        #disease_prob(st.session_state.patient_data)

    


    #print('Accuracy on test data: {:.1f}%'.format(accuracy_score(y_test, y_test_pred)*100))
#    probabilities = full_pipeline.predict_proba(patient_input)
#    return probabilities

st.button('Reset', on_click = reset)
