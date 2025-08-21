from joblib import load
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import os 
import searborn as sns

import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Super store", page_icon=":bar_chart:", layout="wide")

#os.chdir(r"C:\Users\User\Desktop\data analysis\personnel")
df=pd.read_csv("cancer patient data sets.csv")
 
st.sidebar.title("Navigation")
page = st.sidebar.radio("choisissez votre page :", ("Accueil","Analyse","Prediction","Traitement"))

#condition de filtrage
gender = st.sidebar.multiselect("choose gender", df["Gender"].unique(), default=df["Gender"].unique() )
if not gender :
    df2= df.copy()
else:
    df2= df[df["Gender"].isin(gender)].copy()

age_min = int(df2["Age"].min())
age_max = int(df2["Age"].max())

#slider dans la sidebar
age_range = st.sidebar.slider("selectionner une tranche d'age", min_value=age_min, max_value=age_max, value=(age_min, age_max))
if not age_range :
    df3= df2.copy()
else:
    df3= df2[(df2["Age"]>=age_min) & (df2["Age"]<=age_max)]

# filtrge du df
if not gender and not age_range :
    filter_df=df.copy()
elif not gender:
    filter_df= df[(df["Age"]>=age_range[0]) & (df["Age"]<=age_range[1])]
elif not age_range:
    filter_df= df[df["Gender"].isin(gender)]
else:
    filter_df= df[(df["Gender"].isin(gender)) & (df["Age"]>=age_range[0]) & (df["Age"]<=age_range[1])]



if page == "Accueil":
    st.title("Descriptive Analysis")

    col, col2=st.columns((2))
    count= filter_df["Gender"].value_counts().reset_index()
    count.columns=["Gender", "Count"]

    with col:
        st.subheader("nombre de personne par sexe")
        fig = px.bar(count, x="Gender", y="Count", color="Gender", text="Count", labels= {"Gender":"sexe", "Count":"Nombre de personne"})
        #fig.update_traces(text=count, textPosition="outside")
        st.plotly_chart(fig, use_continer_width=True, heigth=200)

    with col2:
        st.subheader("Repartition suivant le sexe")
        fig = px.violin(filter_df, x="Gender", y="Age",color="Gender", color_discrete_sequence=px.colors.qualitative.Set2, box=True, points="all", labels={"Gender":"sexe", "Age":"Age"})
        st.plotly_chart(fig, use_container_width=True, heigth=200)
    
    
    col1,col2=st.columns((2))
    count1= filter_df["Alcohol use"].value_counts().reset_index()
    count1.columns=["Alcohol use", "Count"]
    with col1:
        st.subheader("repartition de l'obesite selon l'age")
        fig= px.violin(filter_df, x="Obesity", y="Age", box=True, points="all", labels={"Obesity":"Obesite", "Age":"Age"})
        fig.update_traces(opacity=0.6)
        st.plotly_chart(fig, use_container_width=True, heigth=200)

    with col2:
        st.subheader("repartition de la quantite d'alcool par niveau")
        fig= px.bar(count1,x="Alcohol use", y="Count",color="Alcohol use", labels={"Alcohol use":"niveau d'alcool", "Count":"quantite"})
        st.plotly_chart(fig, use_container_width=True, heigth=200)

    
    col,col1=st.columns((2))
    count2=filter_df["Smoking"].value_counts().reset_index()
    count2.columns=["Smoking","Count"]
    mean= filter_df.groupby(["Smoking"], as_index=False)["Age"].mean()

    with col:
        st.subheader("Quantite de fumeur par niveau")
        fig= px.bar(count2, x="Smoking", y="Count", labels={"Smoking":"fumeur","count":"Quantite"})
        st.plotly_chart(fig, use_container_width=True, heigth=200)

    with col1:
        st.subheader("moyenne d'age des consommateur")
        fig= px.bar(mean, x="Smoking", y="Age", color_discrete_sequence=px.colors.qualitative.Set2, labels={"Smoking":"fumeur", "Age":"Age"})
        st.plotly_chart(fig, use_container_width=True, heigth=200)

    col,col2= st.columns((2))
    count3=filter_df["Passive Smoker"].value_counts().reset_index()
    count3.columns=["Passive Smoker","Count"]
    mean2= filter_df.groupby(["Passive Smoker"],as_index=False)["Age"].mean()

    with col:
        st.subheader("Quantite de fumeur passif par niveau")
        fig= px.bar(count3, x="Passive Smoker", y="Count", labels={"Passive Smoker":"fumeur passif","count":"Quantite"})
        st.plotly_chart(fig, use_container_width=True, heigth=200)

    with col2:
        st.subheader("moyenne d'age des consommateur")
        fig= px.bar(mean2, x="Passive Smoker", y="Age", color_discrete_sequence=px.colors.qualitative.Set2, labels={"Smoking":"fumeur", "Age":"Age"})
        st.plotly_chart(fig, use_container_width=True, heigth=200)

elif page=="Analyse":

    col,col1,col2= st.columns((3))
    count= filter_df["chronic Lung Disease"].value_counts().reset_index()
    count.columns=["chronic Lung Disease","Count"]

    with col:
        st.subheader("influence de la pollution de l'air sur les maladies chronique")
        fig= px.scatter(filter_df, x="chronic Lung Disease", y="Air Pollution", color_discrete_sequence= px.colors.qualitative.Dark2, labels={"chronic Lung Disease":"maladie chronique","Air Pollution":"pollution de l'air"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col1:
        st.subheader("repartition des maladie chroniue par niveau")
        fig= px.bar(count, x="chronic Lung Disease", y="Count", color_discrete_sequence=px.colors.qualitative.Set1, labels={"chronic Lung Disease":"maladie chronique","Count":"quantite"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("repartition des maladies chronique selon le niveu de tabac")
        fig= px.violin(filter_df,x="chronic Lung Disease", y="Smoking", color_discrete_sequence=px.colors.qualitative.Pastel, labels={"chronic Lung Disease":"maladie chronique", "Smoking":"Fumeur"})
        st.plotly_chart(fig, use_container_width=True)

    cor=filter_df.corr(numeric_only=True)
    
    fig= px.imshow(cor, text_auto=True, color_continuous_scale="Viridis", title="heatmap de correlation")
    fig.update_layout(width=900, height=900, font= dict(size=150), margin=dict(l=30, r=30, t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    col,col1,col2=st.columns((3))
    conso= filter_df.groupby(["Obesity"], as_index=False)["Alcohol use"].mean()
    conso1= filter_df.groupby(["Obesity"], as_index=False)["Smoking"].mean()

    with col:
        st.subheader("consommtion moyenne d'alcool selon l'obesite")
        fig= px.bar(conso, x="Obesity", y="Alcohol use", color_discrete_sequence=px.colors.qualitative.Pastel , labels={"Obesity":"obesite", "Alcohol use":"niveu d'alcool"})
        st.plotly_chart(fig, use_container_width=True)

    with col1:
        st.subheader("repartition de l'obesite selon le niveau d'alcool")
        fig= px.violin(filter_df, x="Alcohol use", y="Obesity", color_discrete_sequence= px.colors.qualitative.Dark2, box=True, points="all", labels={ "Alcohol use":"niveu d'alcool","Obesity":"obesite"})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("influence du tabac sur l'obesite")
        fig=px.scatter(conso1, y="Smoking", x="Obesity", color_discrete_sequence=px.colors.qualitative.Set1, labels={"Smoking":"tabac", "Obesity":"obesite"})
        st.plotly_chart(fig, use_container_width=True)

    col,col1,col2=st.columns((3))
    conso2= filter_df.groupby(["Obesity"], as_index=False)["Balanced Diet"].mean()
    conso3= filter_df.groupby(["Passive Smoker"], as_index=False)["Balanced Diet"].mean()
    conso4= filter_df.groupby(["Obesity"], as_index=False)["Coughing of Blood"].mean()

    with col:
        st.subheader("influence de la mauvaise alimentation sur l'obesite")
        fig=px.bar(conso2, y="Balanced Diet", x="Obesity", color_discrete_sequence=px.colors.qualitative.Set1, labels={"Smoking":"tabac", "Obesity":"obesite"})
        st.plotly_chart(fig, use_container_width=True)

    with col1:
        st.subheader("moyenne d'obesite selon le niveau de tabac")
        fig=px.line(conso3, y="Balanced Diet", x="Passive Smoker", color_discrete_sequence=px.colors.qualitative.Set1, labels={"Smoking":"tabac", "Obesity":"obesite"})
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("influence de l'obesite sur la sante propre")
        fig= px.bar(conso4, y="Coughing of Blood", x="Obesity", color_discrete_sequence=px.colors.qualitative.Dark2, labels={"Obesity":"obesite", "Coughing of Blood":"toux sanglante"})
        st.plotly_chart(fig, use_container_width=True)

    col4,col5=st.columns((2))
    influ=filter_df.groupby(["Obesity"], as_index=False)["Genetic Risk"].mean()
    influent=filter_df.groupby(["chronic Lung Disease"]).agg(moy_alcool=("Alcohol use","mean"), moy_genetique=("Genetic Risk","mean"), douleur_thoraciq=("Chest Pain","mean")).reset_index()
    df_filt= influent.melt(id_vars="chronic Lung Disease", value_vars=["moy_alcool","moy_genetique","douleur_thoraciq"], var_name="categorie", value_name="influence")

    with col4:
        st.subheader("influence des risque genetique sur l'obesite")
        fig= px.bar(influ, x="Obesity", y="Genetic Risk", color_discrete_sequence=px.colors.qualitative.Pastel , labels={"Obesity":"obesite", "Genetic Risk":"Risque genetique"})
        st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.subheader("influence du niveau d'alcool, risque genetique et douleur thoracique sur les maladies chronique pulmonaire")
        fig= px.line(df_filt, x="chronic Lung Disease", y="influence", color="categorie", color_discrete_sequence=px.colors.qualitative.Set1 , labels={"chronic Lung Disease":"maladie chronique pulmonaire", "influence":"facteur margeur"})
        st.plotly_chart(fig, use_container_width=True)
    
elif page=="Prediction":
    model= load('logistic.joblib')

    st.title("Formulaire du patient pour la prediction")

    alcool= st.selectbox("niveau de consommation d'alcool", filter_df["Alcohol use"].unique().tolist()+[0])
    alimentation= st.selectbox("niveau d'alimentation equilibre:", filter_df["Balanced Diet"].unique().tolist()+[0])
    fumeur= st.selectbox("niveau de consommation du tabac:", filter_df["Smoking"].unique().tolist()+[0])
    allergy= st.selectbox("niveau d'allergie a la poussiere: ", filter_df["Dust Allergy"].unique().tolist()+[0])
    maladie= st.selectbox("Niveau de maladie pulmonire chronique :", filter_df["chronic Lung Disease"].unique().tolist()+[0])
    air= st.selectbox("niveau d'exposition a la pollution de l'air: ", filter_df["Air Pollution"].unique().tolist()+[0])
    essouflement= st.selectbox("Niveau d'essouflement: ", filter_df["Shortness of Breath"].unique().tolist()+[0])
    thoracique= st.selectbox("Niveau de douleur thoracique :", filter_df["Chest Pain"].unique().tolist()+[0])
    toux= st.selectbox("Niveau de toux sanglante :", filter_df["Coughing of Blood"].unique().tolist()+[0])
    risque_genetique= st.selectbox("Niveau de risque genetique :", filter_df["Genetic Risk"].unique().tolist()+[0])
    passif= st.selectbox("Niveau d'exposition en tant que fumeur passif ", filter_df["Passive Smoker"].unique().tolist()+[0])
    fatigue= st.selectbox("Niveau de fatigue observe :", filter_df["Fatigue"].unique().tolist()+[0])
    obesite=st.selectbox("Niveau d'obesite :", filter_df["Obesity"].unique().tolist()+[0])
    risque_prof= st.selectbox("Niveau de risque professionnel :", filter_df['OccuPational Hazards'].unique().tolist()+[0])

    if st.button("probabilite d'obtention du cancer"):
        data= np.array([[alcool,alimentation,fumeur,allergy,essouflement,thoracique,maladie,air,toux,risque_genetique,passif,risque_prof,fatigue,obesite]])

        predict= model.predict_proba(data)[0][1]*100

        st.success(f"la probabilite d'avoir le cancer est de {predict:.2f}%")












