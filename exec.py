
import os
import pandas as pd
import matplotlib.pyplot as plt
from getpass import getpass
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

api_key = st.text_input("Enter your OpenAI API Key:", type="password")
os.environ['OPENAI_API_KEY'] = api_key


df = pd.read_csv("customers.csv")

def display():
    st.write(df.head())

def schema():
    st.write(df.info())


data_analysis_agent = create_pandas_dataframe_agent(OpenAI(temperature=0),df,verbose=True)

def analyze():
    st.write(data_analysis_agent.invoke("Analyze this data."))

def query():
    prompt = st.text_input("Enter your query")
    if st.button('Submit Query'):
        if prompt:
            result = data_analysis_agent.run(prompt)
            st.write(result)
        else:
            st.write("Please enter a query.")


if st.button('Display Data'):
    display()

if st.button('Display Schema'):
    schema()

if st.button('Analyze Data'):
    analyze()

query()
