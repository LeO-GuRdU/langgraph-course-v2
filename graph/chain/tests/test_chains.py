from pprint import pprint
from dotenv import load_dotenv
from graph.chain.generation import generation_chain
# from graph.chain.hallucination_grader import (GradeHallucinations, hallucination_grader)
from graph.chain.retrieval_grader import GradeDocuments, retrieval_grader
# from graph.chain.router import RouteQuery, question_router
from ingestion import retriever

load_dotenv()

def test_foo() -> None:
    assert 1 == 1

def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"

def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)