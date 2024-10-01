from fastapi import APIRouter
from src.tools.startup import params, logger
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
    Answers the given query question using the RAG agent.

    Args:
        query(str): The question to be answered

    Returns:
        answer(JSON): The answer to the question
    """

    logger.info(f"Started answering question '{query}'")
    answer = rag.answer_question(query).split("\n")[-1]
    logger.info("Answered question")

    return {"answer": answer}
