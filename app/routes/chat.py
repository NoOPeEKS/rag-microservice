from fastapi import APIRouter
from app.models import example as example_models


router = APIRouter(
    prefix="/chat",
)

@router.post(
    path="/",
    response_model=example_models.OutputBody,
    responses={
        200: {"Description": "Example post method"}})
def answer_query(body: example_models.InputBody) -> example_models.OutputBody:
    """
    Uses RAG to answer the given query
    """

    return {"answer": "fuck you"}



