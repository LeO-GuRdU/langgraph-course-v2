from typing import Dict, Any
from graph.state import GraphState
from graph.chain.generation import generation_chain

def generate(state: GraphState) -> Dict[str, Any]:
    print("--- Generating answer based on the question and documents ---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}