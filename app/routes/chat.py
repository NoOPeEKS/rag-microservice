from fastapi import APIRouter, Request
from src.tools.startup import logger


router = APIRouter(
    prefix="/chat",
)


@router.post(
    path="/ask",
    responses={
        200: {"Description": "The answer to the user requested question"}})
def answer_query(query: str, request: Request):
    """
    Answers the given query question using the RAG agent.

    Args:
        query(str): The question to be answered

    Returns:
        answer(JSON): The answer to the question
    """

    rag = request.app.rag
    logger.info(f"Started answering question '{query}'")
    answer = rag.answer_question(query).split("\n")[-1]
    logger.info("Answered question")

    return {"answer": answer}
