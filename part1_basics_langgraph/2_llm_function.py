import os
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict


model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 1. Define the state
class State(TypedDict):
    message: str

def get_response_from_llm(state):
    """
    This function will get the response from the LLM
    
    """

    state["message"]  = state.get("message", "")

    response = model.invoke(state["message"]).content

    return {"message": response}


def convert_to_uppercase(state):
    """
    This function will convert the message to uppercase
    """
    response_from_llm = state["message"]
    state["message"] = response_from_llm.upper()
    
    return state

# define the workflow

workflow = StateGraph(State)
workflow.add_node("LLM_response" , get_response_from_llm)
workflow.add_node("Uppercase_converter" , convert_to_uppercase)
workflow.add_edge(START , "LLM_response")
workflow.add_edge("LLM_response" , "Uppercase_converter")
workflow.add_edge("Uppercase_converter" , END)
app = workflow.compile()


if __name__ == "__main__":
    result = app.invoke({"message": "Hello, how are you?"})
    print(result["message"])