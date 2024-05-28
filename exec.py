
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
            
            # Example query processing for generating graphs
            if "plot" in prompt.lower() and "bar" in prompt.lower():
                # Generate a bar plot
                x_label = st.text_input("Enter x-axis label")
                y_label = st.text_input("Enter y-axis label")
                plot_data = df[prompt.split('from')[1].split('to')[0]]
                fig, ax = plt.subplots()
                plot_data.plot(kind='bar', ax=ax)
                ax.set_title('Bar Plot')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                st.pyplot(fig)  # Display the plot
            elif "plot" in prompt.lower() and "line" in prompt.lower():
                # Generate a line plot
                x_label = st.text_input("Enter x-axis label")
                y_label = st.text_input("Enter y-axis label")
                plot_data = df[prompt.split('from')[1].split('to')[0]]
                fig, ax = plt.subplots()
                plot_data.plot(kind='line', ax=ax)
                ax.set_title('Line Plot')
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                st.pyplot(fig)  # Display the plot
            elif "plot" in prompt.lower() and "histogram" in prompt.lower():
                # Generate a histogram
                x_label = st.text_input("Enter x-axis label")
                plot_data = df[prompt.split('from')[1].split('to')[0]]
                fig, ax = plt.subplots()
                plot_data.plot(kind='hist', ax=ax)
                ax.set_title('Histogram')
                ax.set_xlabel(x_label)
                st.pyplot(fig)  # Display the plot
            else:
                st.write("No specific graph was generated for the query.")
        else:
            st.write("Please enter a query.")


if st.button('Display Data'):
    display()

if st.button('Display Schema'):
    schema()

if st.button('Analyze Data'):
    analyze()

query()
