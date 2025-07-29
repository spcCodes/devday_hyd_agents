# Part 1: Basics of LangGraph

This folder contains introductory examples and a notebook to help you get started with the basics of building workflows using Langgraph

## Contents

- **1_simple_workflow.py**  
  Demonstrates how to define a simple workflow graph with two nodes:
  1. `greet_user`: Greets the user using their input.
  2. `convert_to_uppercase`: Converts the greeting message to uppercase.
  
  The workflow is defined using `StateGraph`, and the script shows how to invoke the workflow with a sample input.

- **2_llm_function.py**  
  Extends the previous example by integrating an LLM (OpenAI GPT-4.1-mini) into the workflow:
  1. `get_response_from_llm`: Sends the user's message to the LLM and stores the response.
  2. `convert_to_uppercase`: Converts the LLM's response to uppercase.
  
  This script demonstrates how to use an LLM as a node in a LangGraph workflow.

- **session1_basics_langgraph.ipynb**  
  An interactive Jupyter notebook that walks through:
  - Setting up the environment and loading dependencies.
  - Defining utility functions for displaying graphs and streaming output.
  - Step-by-step construction of simple workflow graphs.
  - Integrating LLMs into workflows.
  - Visualizing and experimenting with the workflow graph interactively.

## Getting Started


1. **Run the scripts**  

   - To run the simple workflow:

     ```
        python part1_basics_langgraph/1_simple_workflow.py
     ```

   - To run the LLM workflow:
     ```
        python part1_basics_langgraph/2_llm_function.py
     ```

2. **Explore the notebook**  
   - Open `session1_basics_langgraph.ipynb` in Jupyter or VSCode to interactively explore and modify the workflows.

## Concepts Demonstrated

- **State Management:** Using Python dictionaries and `TypedDict` to manage workflow state.
- **Workflow Graphs:** Building and connecting nodes using LangGraph's `StateGraph`.
- **Node Functions:** Creating modular functions for each step in the workflow.
- **LLM Integration:** Calling an LLM as part of a workflow node.
- **Visualization:** (In the notebook) Visualizing the workflow graph and streaming outputs.


