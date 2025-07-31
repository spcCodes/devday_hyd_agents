import os 
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph , START , END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage , AIMessage , SystemMessage
from typing import TypedDict , Annotated
from pydantic import BaseModel , Field
from typing import Literal
from langgraph.graph import StateGraph, END
import json



class GraphState(TypedDict):
    messages: Annotated[list, add_messages]
    text_classification: Annotated[str, None]
    entity_extraction: Annotated[dict[str, list[str]], None]
    summary: Annotated[str, None]
    title: Annotated[str, None]
    content: Annotated[str, None]


model = ChatOpenAI(model="gpt-4.1-mini", temperature=0 , max_completion_tokens=500)



# Pydantic Schemas and LLMs
class TextClassification(BaseModel):
    """
    Classify the text into one of the following categories: News, Blog, Research, or Other.
    """
    category: Literal["News", "Blog", "Research", "Other"] = Field(description="The category of the text")

text_classification_llm = model.with_structured_output(TextClassification)

class EntityExtraction(BaseModel):
    """
    Extract entities from the text. Return a dictionary with keys 'person', 'organisation', and 'location'.
    Each key should map to a list of the corresponding entities found in the text.
    Example: {"person": ["John Doe"], "organisation": ["OpenAI"], "location": ["San Francisco", "India"]}
    If no entities are found for a category, return an empty list for that key.
    """
    person: list[str] = Field(default=[], description="List of person names found in the text")
    organisation: list[str] = Field(default=[], description="List of organization names found in the text") 
    location: list[str] = Field(default=[], description="List of location names found in the text")

entity_extraction_llm = model.with_structured_output(EntityExtraction)

class TextSummarization(BaseModel):
    """
    Summarize the text in a concise manner.
    """
    summary: str = Field(description="The summary of the text")

text_summarization_llm = model.with_structured_output(TextSummarization)

class TitleGeneration(BaseModel):
    """
    Generate the title of the text.
    """
    title: str = Field(description="The title of the text")

title_generation_llm = model.with_structured_output(TitleGeneration)

class ContentGeneration(BaseModel):
    """
    Join the content of the text above and generate the content of the text.
    """
    content: str = Field(description="The content of the text")

content_generation_llm = model.with_structured_output(ContentGeneration)

# --- LangGraph Node Functions ---

def classify_text(state):
    print('-> Calling text classification agent ->')

    text_classification_system_prompt = """
    You are a text classification agent. You will be given a user query and you will need to classify the text into one of the following categories: News, Blog, Research, or Other.

    Respond in json format with the following keys:

    category: The category of the text. Must be one of "News", "Blog", "Research", or "Other".
    """

    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=text_classification_system_prompt)]

    messages = system_message + ai_message + human_message

    result = text_classification_llm.invoke(messages)

    # We only update the text_classification, not messages. The router will use the category.
    return {
        "text_classification": result.category
    }

def extract_entities(state):
    print("-> Calling entity extraction agent ->")
    entity_extraction_system_prompt = """
    You are an entity extraction agent. Extract entities from the provided text. 
    Return a JSON object with keys 'person', 'organisation', and 'location'.
    Each key should map to a list of the corresponding entities found in the text.
    
    - person: List of people's names (e.g., ["John Doe", "Jane Smith"])
    - organisation: List of company/organization names (e.g., ["OpenAI", "Microsoft"])  
    - location: List of places/locations (e.g., ["San Francisco", "India"])
    
    If no entities are found for a category, return an empty list for that key.
    """
    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=entity_extraction_system_prompt)]
    messages = system_message + ai_message + human_message
    
    result = entity_extraction_llm.invoke(messages)
    
    # Create the entities dictionary from the direct fields
    entities_dict = {
        "person": result.person,
        "organisation": result.organisation, 
        "location": result.location
    }
    
    
    return {
        "entity_extraction": entities_dict
    }

def summarize_text(state):
    print("-> Calling text summarization agent ->")
    
    # Define the system prompt for summarization
    text_summarization_system_prompt = """
    You are a text summarization agent. Summarize the following text in 2-3 sentences, 
    focusing on the main points and omitting unnecessary details. 
    Be concise and clear.
    """
    
    # Get the user's input text from messages
    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=text_summarization_system_prompt)]
    
    # Prepare messages for the LLM
    messages = system_message + ai_message + human_message
    
    # Use the structured LLM to get the summary
    result = text_summarization_llm.invoke(messages)
    
    
    return {"summary": result.summary}

def generate_title(state):
    print("🏷️ Generating title...")
    
    # Define the system prompt for title generation
    title_generation_system_prompt = """
    You are a title generation agent. Generate a compelling and relevant title for the given text.
    The title should capture the main theme and be engaging for readers.
    """
    
    # Get the user's input text from messages
    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=title_generation_system_prompt)]
    
    # Prepare messages for the LLM
    messages = system_message + ai_message + human_message
    
    # Use the structured LLM to get the title
    result = title_generation_llm.invoke(messages)
    
    
    return {"title": result.title}

def generate_content(state):
    print("📰 Generating content...")
    
    # Compose content generation prompt using previous results
    summary = state.get("summary", "")
    entities = state.get("entity_extraction", {})
    category = state.get("text_classification", "")
    title = state.get("title", "")
    
    # Define the system prompt for content generation
    content_generation_system_prompt = f"""
    You are a content generation agent. Based on the following information, write a comprehensive blog post:
    
    Category: {category}
    Title: {title}
    Summary: {summary}

    Write a well-structured blog post that just concantenates the information above. 

    The blog post should be in the following format:

    <h1>Category</h1>
    <h1>Title</h1
    <p>Summary</p>


    dont make any cosmetic chnages but put it in a well strucvtured manner.
  
    """
    
    # Get the user's input text from messages
    human_message = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    ai_message = [msg for msg in state["messages"] if isinstance(msg, AIMessage)]
    system_message = [SystemMessage(content=content_generation_system_prompt)]
    
    # Prepare messages for the LLM
    messages = system_message + ai_message + human_message
    
    # Use the structured LLM to get the content
    result = content_generation_llm.invoke(messages)

    
    return {"content": result.content}


# --- LangGraph Workflow Construction ---

# Workflow
# classify_text -> extract_entities -> summarize_text -> generate_title -> generate_content -> END

graph = StateGraph(GraphState)
# Add nodes for each step
graph.add_node("classify_text", classify_text)
graph.add_node("extract_entities", extract_entities)
graph.add_node("summarize_text", summarize_text)
graph.add_node("generate_title", generate_title)
graph.add_node("generate_content", generate_content)

# Define the workflow edges
graph.add_edge("classify_text", "extract_entities")
graph.add_edge("extract_entities", "summarize_text")
graph.add_edge("summarize_text", "generate_title")
graph.add_edge("generate_title", "generate_content")
graph.add_edge("generate_content", END)

# Set the entry point
graph.set_entry_point("classify_text")

# Compile the workflow app
awesome_langgraph_workflow = graph.compile()


# --- Example: Run the workflow on a sample paragraph ---

# Example input paragraph (longer version with clear entities)
example_paragraph = (
    "OpenAI CEO Sam Altman announced at a press conference in San Francisco yesterday that the company "
    "has secured a groundbreaking partnership with Microsoft to advance artificial intelligence research. "
    "The collaboration, worth $10 billion, will establish new AI research centers in Seattle, London, and Tokyo. "
    "Leading researchers from Stanford University, including Dr. Fei-Fei Li and Dr. Andrew Ng, will join "
    "the initiative alongside teams from Google DeepMind and Meta AI. The project aims to develop next-generation "
    "language models that can assist in solving climate change challenges across regions like Sub-Saharan Africa "
    "and Southeast Asia. Satya Nadella, Microsoft's CEO, emphasized during the announcement in Redmond that "
    "this partnership represents a significant step forward in democratizing AI technology. The initiative will "
    "also involve collaboration with the European Union's AI research consortium and the Singapore government's "
    "Smart Nation program. Industry experts believe this development could revolutionize how AI is deployed "
    "in developing countries, potentially impacting millions of lives in India, Brazil, and Nigeria. "
    "The research will focus on creating AI systems that can operate efficiently in low-resource environments "
    "while maintaining high accuracy and ethical standards."
)

# Prepare the initial state for the workflow
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [HumanMessage(content=example_paragraph)]
}

# Run the workflow
result = awesome_langgraph_workflow.invoke(initial_state)

# Access individual values
classification = result['text_classification']
entities = result['entity_extraction']
summary = result['summary']
title = result['title']
content = result['content']

# At the end, create a JSON-serializable result
json_result = {
    'text_classification': result.get('text_classification'),
    'entity_extraction': result.get('entity_extraction'),
    'summary': result.get('summary'),
    'title': result.get('title'),
    'content': result.get('content'),
    'original_text': result['messages'][0].content if result.get('messages') else None
}

# Convert to JSON string
json_output = json.dumps(json_result, indent=2, ensure_ascii=False)
print("JSON OUTPUT:")
print(json_output)

# Save to file if needed
with open('workflow_result.json', 'w', encoding='utf-8') as f:
    json.dump(json_result, f, indent=2, ensure_ascii=False)