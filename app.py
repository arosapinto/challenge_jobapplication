import random
import os

import streamlit as st
import streamlit.components.v1 as components
import lime
from lime import lime_tabular
import pandas as pd
import numpy as np
import joblib as jl

random.seed(10)

###page configuration
st.set_page_config(page_title="Tech challenge", page_icon=None, layout="wide", initial_sidebar_state="auto", menu_items=None)
st.title("Understand your black box algorithm with LIME") # corresponds to H1 heading
 ### app explanaiton
st.write("This app was built to provide a platform to explain the prediction performed by a linear regression model.\
         LIME (Local interpretable model-agnostic explanations) algorithm explains a black-box model by explaining the relationship between the input and the model’s prediction.\
             The user has to provide the training data set, trained BlackBox model and one sample(corresponding to user input) to be explained by LIME.\
                     More information about LIME can be found in https://arxiv.org/abs/1602.04938.")

###side bar creation
st.sidebar.title("Upload your csv and the model") 
#upload data;  model and select model type
upload_data=st.sidebar.file_uploader("Upload csv", type=["csv"])
choices = {
    1: "Linear Regression",
    2: "Recurrent neural networks"
}

def format_func(option):
    return choices[option]

option = st.sidebar.selectbox("What's your model type?", options=list(choices.keys()), format_func=format_func)

st.sidebar.write(f"You selected option {option} called {format_func(option)}")

upload_model=st.sidebar.file_uploader("Upload model", type=["pkl"])
   
if (upload_data and upload_model):
    df = pd.read_csv(upload_data)
    #st.write(f"Dataset uploaded")
    #st.write(df)
    load_model=jl.load(upload_model)

    # features and target definition, target has to be the last column of the dataset
    features = df.columns[:-1]
    num_features = len(features)
    target = df.columns[-1]
    xtrain = np.array(df[features])
    ytrain = np.array(df[target])

    categorical_features = np.argwhere(np.array([len(set(xtrain[:,x])) for x in range(xtrain.shape[1])]) <= 10).flatten()
   
    st.header("Brief summary of the variables")
    ### variables explanation, the 
    for i, col in enumerate(df.columns):
        if i in categorical_features:
            possible_values = df[col].unique()
            st.write(f"{col} - categorical | possible values: {', '.join([str(i) for i in np.sort(possible_values)])}")
        else:
            minimum = df[col].min()
            maximum = df[col].max()
            st.write(f"{col} - numerical | min: {minimum} | max: {maximum}")

####user inputs 
    st.header("Please fill in the required below inputs") # corresponds to H2 heading
    user_inputs = {}
    for col in df.columns[:-1]:
        user_input = st.text_input(col)
        if user_input:
            user_inputs[col] = float(user_input)

    m = st.markdown("""
      <style>
       div.stButton > button:first-child {
           background-color: #006666;
           color:#ffffff;
       }
       div.stButton > button:hover {
           background-color: #bcbcbc;
           color:#ffffff;
           }
       </style>""", unsafe_allow_html=True)
    



    ###Explain the model with LIME
    if st.button("Explain"):
        user_inputs_list = [user_inputs]
        model_type = format_func(option)
        sample_data_1 = pd.DataFrame(user_inputs_list)
        sample_data_1 = sample_data_1.values.reshape(-1)

        if model_type == "Linear Regression":
            explainer = lime_tabular.LimeTabularExplainer(
                xtrain,
                feature_names=list(features),
                class_names=["charges"],
                categorical_features=categorical_features,
                verbose=True,
                mode="regression"
            )               
            exp = explainer.explain_instance(sample_data_1, load_model.predict, num_features=num_features)
        elif model_type == "Recurrent neural networks":
            st.write("You selected recurrent neural networks, the website isn't ready for addressing this model yet")
            explainer_nn = lime_tabular.RecurrentTabularExplainer(
                xtrain,
                training_labels=categorical_features,
                feature_names=list(features),
                discretize_continuous=True,
                class_names=["charges"],
                discretizer="decile"
            )
            exp = explainer_nn.explain_instance(sample_data_1, load_model.predict, num_features=num_features)
        else:
            st.text("Not implemented")

        exp.as_list()
        exp.save_to_file("lime.html")

        with open("lime.html", "r", encoding="utf-8") as html:
            source_code = html.read()

        st.header("LIME results")
        components.html(source_code)

        os.remove("lime.html")
