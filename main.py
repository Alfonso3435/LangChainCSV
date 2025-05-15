import streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os



load_dotenv()
def getOpenAIkey():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    return OPENAI_API_KEY

api_key = getOpenAIkey()

llm = ChatOpenAI(model="gpt-4", temperature=0, api_key=api_key)
st.title("CSV CHAT APP")
st.write("Upload your CSV and ask anything")
file = st.file_uploader("Select your file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    input = st.text_area("Ask your question here")
    if input is not None:
        button = st.button("Submit")
        agent = create_pandas_dataframe_agent(
            llm, df, verbose=False, agent_type=AgentType.OPENAI_FUNCTIONS, allow_dangerous_code=True
        )

        if button:
            result = agent.invoke(input)
            st.write(result["output"])
