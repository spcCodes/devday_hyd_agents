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

#nodes definition
def greet_user(state: State) -> State:
    """Greet the user with a message."""
    user_message = state.get("message", "")
    return {"message": f"Hello {user_message}!"}


def convert_to_uppercase(state: State) -> State:
    """Convert the input word in the state to uppercase."""
    state["message"] = state["message"].upper()
    return state

#define the workflow

workflow = StateGraph(State)
workflow.add_node("User_greetings", greet_user)
workflow.add_node("Uppercase_converter", convert_to_uppercase)
workflow.add_edge(START, "User_greetings")
workflow.add_edge("User_greetings", "Uppercase_converter")
workflow.add_edge("Uppercase_converter", END)
app = workflow.compile()

if __name__ == "__main__":
    result = app.invoke({"message": "Suman"})
    print(result)