import os
from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

class State(TypedDict):
    messages:Annotated[list,add_messages]

# To stream the output of the app
def stream_output(app, input):
    for output in app.stream(input):
        for key,value in output.items():
            print(f"here is output from {key}")
            print("_______")
            print(value)
            print("\n")

def classify_intent(state:State):
    """
    This function will classify the intent of the message
    """
    user_input = state["messages"][0].content
    prompt = f"You have to classify the intent of the message. this can be either question, feedback, help, complaint, or other. Return the intent as a string."
    final_message = user_input + prompt
    response = model.invoke(final_message).content
    return {"messages": [response]}

def respond_to_intent(state:State):
    """
    This function will respond to the intent of the message
    """
    user_input = state["messages"][0].content
    intent = state["messages"][-1].content

    prompt = "You are an intelligent assistant. You will be given an user input and intent. you jave to respond to the user based on the intent. Return the response as a string."
    final_message = user_input + intent + prompt
    response = model.invoke(final_message).content
    return {"messages": [response]}


graph = StateGraph(State)
graph.add_node("classify_intent", classify_intent)
graph.add_node("respond_to_intent", respond_to_intent)

graph.add_edge(START, "classify_intent")
graph.add_edge("classify_intent", "respond_to_intent")
graph.add_edge("respond_to_intent", END)

app = graph.compile()

if __name__ == "__main__":
    # result = app.invoke({"messages": "I have to go to citycenter mall. can you tell me the shortest route to reach there?"})
    result = app.invoke({"messages": "I am not happy with the product. I want to return it."})
    print(result)

    # stream_output(app, {"messages": "I am not happy with the product. I want to return it."})