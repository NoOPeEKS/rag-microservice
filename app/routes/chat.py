from fastapi import APIRouter
from src.tools.startup import params
from src.models.rag import RAG


rag = RAG(params['global']['rag_chatbot'])
router = APIRouter(
    prefix="/chat",
)

@router.post(
    path="/ask",
    responses={
        200: {"Description": "The answer to the user requested question"}})
def answer_query(query: str):
    """
    Uses RAG to answer the given query
    """

    answer = rag.answer_question(query).split("\n")[-1]

    return {"answer": answer}



