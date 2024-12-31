import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

def getLLamaresponse(input_text,no_words,blog_style):
    llm=CTransformers(model='models\llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type="llama",
                      config={'max_new_tokens':int(no_words) + 20,
                              'temperature':0.01})
    
    template="""
    Write a blog for {blog_style} job profile for a topic {input_text} within {no_words} words"""

    prompt=PromptTemplate(input_variables=["blog_style","input_text","no_words"],template=template)

    resp=llm(prompt.format(blog_style=blog_style,input_text=input_text,no_words=no_words))
    print(resp)
    return resp
st.set_page_config(page_title="Generate blogs",page_icon=":)",layout='centered',initial_sidebar_state="collapsed")
st.header("Generate Blog")
input_text=st.text_input("Enter blog topic")

col1,col2=st.columns([5,5])
with col1:
    no_words=st.text_input('No of words')
with col2:
    blog_style=st.selectbox("Target Audience",('Researchers','Data Scientist','Common people'),index=0)

submit=st.button("Generate")

if submit:
    st.write(getLLamaresponse(input_text,no_words,blog_style))
