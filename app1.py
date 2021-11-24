# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 21:10:14 2021

@author: admin
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 19:23:05 2021

@author: admin
"""
import streamlit as st
import pandas as pd
import pickle


def main():
    st.title("Home")
    html_temp = """
    <div style="background-color:black;padding:15px">
    <h2 styles="color:white;text-align:left;"> Fumble </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
if __name__=='__main__':
    main()   
        
#stored the dataset into cache memory    
@st.cache()
def data():
    okc=pd.read_excel("D:/5th Sem/Project/Copy of User Details_Faf.xlxs") 
    return data


#when we receive a response, tat will be appened to the cache memory    
@st.cache()
def get_data():
    return []

name = st.text_input("Name","Type..")

age = st.number_input("Age",18,79)

gender = st.selectbox("Gender", ["Female","Male"])

oreintation = st.selectbox("Orientation", ["Straight","Bisexual","Gay"])

status = st.selectbox("Status", ["Single","in a relationship","Unknown"])

education = st.selectbox("Education", ["college/university","Masters and above","other","Two-year college","High school","Med / Law school"])

ethnicity = st.selectbox("Ethnicity", ["White","Asian","Hispanic","African American","Mixed","Unknown","others"])

religion = st.selectbox("Religion", ["Agnosticism","Atheism","Christianity","Catholicism","Judaism","Buddhism","Islam","Hinduism","Unknown","others"])

smokes = st.selectbox("Smokes", ["Yes","No"])

drink = st.selectbox("Drink", ["Yes","No"])

body_type = st.selectbox("Body_Type", ["Fit","Average","Curvy","Thin","Overweight","Rather not say"])

diet = st.selectbox("Diet", ["Anything","Vegan","Vegetarian","Halal","Kosher","other"])

job = st.selectbox("Job", ["Office/Professional","Science/Tech","Business Management","Creative"])

speaks = st.text_input("Speaks \n Enter your 2nd preferred language","Type..")

sign = st.selectbox("Sign", ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpion","Sagittarius","Capricorns","Aquarius","Pisces"])

offspring = st.selectbox("Offspring", ["Wants Kids","Does not want kids","Has kid","Unknown"])

drugs = st.selectbox("Drugs", ["Yes","No"])

height = st.number_input("Height \n (In inches)",30,100)

income = st.number_input("Income",0,100000)

pets = st.selectbox("Pets", ["Likes Cats and Dogs","Dislikes Cats and Dogs","Likes only cats","Likes only Dogs","Unknown"])

essay0 = st.text_input("essay0 My self summary","Type..")

essay1 = st.text_input("essay1 What I am doing with my life","Type..")

essay2 = st.text_input("essay2 I am really good at ","Type..")

essay3 = st.text_input("essay3 The first thing people usually notice about me","Type..")

essay4 = st.text_input("essay4 Favourite books, Movies, Show,Music, Food","Type..")

essay5 = st.text_input("essay5 The 6 things that I could never do without","Type..")

essay6 = st.text_input("essay6 I spend a lot of time thinking about","Type")

essay7 = st.text_input("essay7 On a typical Friday night I am","Type..")

essay8 = st.text_input("essay8 The most private thing I am willing to admit","Type..")

essay9 = st.text_input("essay9 You should message me if","Type..")

if st.button("Add row"):
    data().append({"Name": name, "Age": age, "Gender": gender, "Orientation": orientation,
                       "Status":status, "Education":education, "Ethnicity":ethnicity,
                       "Religion":religion,"Smokes":smokes,"Drinks":Drinks,
                       "Diet":diet,"Speaks":speaks,"Sign":sign,"Offspring":offspring,
                       "Drugs":drugs,"Height":height,"Income":income,"Job":job,
                       "Pets":pets,"essay0 My self summary":essay0, "essay1 What I am doing with my life":essay1, "essay2 I am really good at":essay2,
                       "essay3 The first thing people usually notice about me":essay3,
                       "essay4 Favourite books, Movies, Show,Music, Food":essay4,
                       "essay5 The 6 things that I could never do without":essay5, 
                       "essay6 I spend a lot of time thinking about":essay6, 
                       "essay7 On a typical Friday night I am":essay7,
                       "essay8 The most private thing I am willing to admit":essay8,
                       "essay9 You should message me if":essay9})
    
    
#st.write(pd.DataFrame(data()))  





  
'''     
     #------------------------------------------
@st.cache ()

def match(name,age,gender,orientation,status,education,ethnicity,religion,
          smokes,drink,body_type,diet,job,speaks,sign,offspring,
          drugs,height,income,pet,essay0,essay1,essay2,essay3,
          essay4,essay5,essay6,essay7,essay8,essay9):
    
    #data preprocessing:
        
        okc=pd.read_excel("D:/5th Sem/Project/Copy of User Details_Faf.xlsx")

        ok = okc.copy(deep=True)

       ok['essay']=ok[['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9']].agg(' '.join, axis=1)

       ok.drop(['essay0','essay1','essay2','essay3','essay4','essay5','essay6','essay7','essay8','essay9','Name'],axis=1,inplace=True)



corpus_df = ok.copy(deep=True)

    
    
   '''  
   

      