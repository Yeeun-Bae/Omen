import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import ast
import seaborn as sns
import matplotlib.pyplot as plt


# Needed to display the different pages
placeholder=st.empty()

## Initialize session state variables for later use
if "page" not in st.session_state:
    # manages the currently displayed pate
    st.session_state.page = 0
if "question" not in st.session_state:
    # manages the currently displayed user query
    st.session_state.question = 0
if "personal" not in st.session_state:
    # saves user inputs
    st.session_state.personal = pd.DataFrame(columns =  ['symptom_desc', 'class_desc'])
if "patient_data" not in st.session_state:
    # holds translated model inputs
    st.session_state.patient_data = pd.DataFrame(columns =  ['disease', 'input']) 

def diagnose():
    # After the initial choice, extract the relevant diseases 
    st.session_state.diseases = set(symptoms[symptoms['chosen']]['diseases'].sum())
    st.session_state.page = 1

def reset():
    # reset session state variables to initial values. Start anew.
    st.session_state.page = 0
    st.session_state.question = 0
    st.session_state.personal = pd.DataFrame(columns =  ['symptom_desc', 'class_desc'])
    st.session_state.patient_data = pd.DataFrame(columns =  ['disease', 'input']) 

def results():
    # display the results page on button click
    st.session_state.page = 2

def next_q(answer, q):
    # After input, save the signifier/value pair to a dataframe and show the next query.
    st.session_state.question +=1
    new  = pd.DataFrame([[q, answer]], columns = ['symptom_desc', 'class_desc'])
    st.session_state.personal = pd.concat([st.session_state.personal, new], axis = 'index', ignore_index=True)

def mk_model_in(dis, df):
    # takes a constructed dataframe of model inputs and the associated disease and saves them for further use
    if ~(dis in st.session_state.patient_data['disease']):
        new = pd.DataFrame([[dis, dict(zip(df['symptom'], df['value']))]], columns = ['disease', 'input'])
        st.session_state.patient_data = pd.concat([st.session_state.patient_data, new], axis = 'index', ignore_index=True)
    

# General symptoms, their group and associated diseases. Used to determine which inputs to query.
symptoms = pd.DataFrame(data=np.array([
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


# First page. Display groups of general symptoms.
if st.session_state.page == 0:
    with placeholder.container():
        for group in symptoms["group"].unique():   
            with st.expander(group):
                for symp in symptoms[symptoms["group"]==group]["name"]:
                    symptoms.loc[symptoms["name"] == symp, 'chosen'] = st.checkbox(symp, False)
        st.button('Diagnosis',on_click=diagnose, type='primary', disabled = ~symptoms['chosen'].any())

# Second page, query user inputs
elif st.session_state.page == 1:
    with placeholder.container():
        # dataframe detailing the relationship between symptoms, diseases, signifiers and human->machine translation
        ds = pd.read_csv(str(Path.cwd())+r'/disease_symptoms.csv')

        #st.write(ds[ds['disease'].isin(st.session_state.diseases)])

        # Query input for every symptom needed.
        for idx, q in enumerate(ds[ds['disease'].isin(st.session_state.diseases)]['symptom_desc'].unique()):
            # displays only one question at a time
            if st.session_state.question == idx:
                st.write(f"Please provide additional information on patient's {q}:")

                # All symptoms that will be updated from this query
                symps = ds[ds['symptom_desc']==q]
                # A value needs to be continuous when even one of them is not categorical
                # Categorical values will a list structure in "class" and thus contain ','
                is_cont = ~symps['class'].str.contains(',').all()
                
                if(is_cont):
                    # Continuous variables list a min-max range in 'class'. 
                    # Combine all of them and take the ultimate min and max from there.
                    thresholds = list(map(float,(symps[(~ds['class'].str.contains(','))]["class"].str.split(' - ').values.sum())))
                    maximum = max(thresholds)
                    minimum = min(thresholds)
                    label = f"{minimum} - {maximum}"

                    # take continous value as free text-input
                    answer = st.text_input(label = label, key = q)
                            
                elif(~is_cont):
                    # Categorical values that are not humanly legible include a dictionary of translations in 'class_desc'
                    translation = ~symps['class_desc'].isna().all()

                    if(translation):
                        options = list(ast.literal_eval(symps[~symps['class_desc'].isna()]['class_desc'].iloc[0]).keys())
                    else:
                        # Take the option with the most categories to display. 
                        # Other options will have a dictionary to translate to the lower number of options
                        here = symps['class'].str.len().nlargest(1).index
                        options = ast.literal_eval(ds['class'].iloc[here].values[0])

                    answer = st.selectbox(label='', options = options, key = q, label_visibility = 'collapsed')

                # Confirm input and get the next question             
                st.button("Next", on_click=next_q, args=[answer, q])               
        
        # Once all the inputs have been made
        if st.session_state.question == len(ds[ds['disease'].isin(st.session_state.diseases)]['symptom_desc'].unique()):
            # display input overview
            st.write(st.session_state.personal)
            # Extract specific inputs for each model
            for dis in st.session_state.diseases:
                # Get only input relevant to the disease
                this = st.session_state.personal.merge(ds[ds['disease']==dis], on='symptom_desc', how='inner')
                
                # If no translation is necessary, take the input directly
                this.loc[this['class_desc_y'].isna(),'value']= this['class_desc_x']
        
                for ii in this[this['bnr'].isna() & ~this['class_desc_y'].isna()].index:
                    # If a translation dictionary is present, translate the human input into correct model label
                    this.iloc[ii, this.columns.get_indexer(['value'])] = eval(this.iloc[ii]['class_desc_y'])[this.iloc[ii]['class_desc_x']]
                for ii in this[~this['bnr'].isna()].index:
                    # If a binary classification from a continuous variable is needed, check against threshold and translate into correct model label
                    this.iloc[ii, this.columns.get_indexer(['value'])] = eval(this.iloc[ii]['class_desc_y'])[float(this.iloc[ii]['class_desc_x']) > float(this.iloc[ii]['bnr'])]

                # Save the extracted values to use in  
                mk_model_in(dis, this)
            
            # Go to next page to show results
            st.button('Show Results',on_click=results, type='primary')             
        
        

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
                    probabilities.append(round(alzheimer_logistic.predict_proba(input)[0][0], 2) * 100)
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
