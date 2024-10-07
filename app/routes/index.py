from fastapi import APIRouter, Request
from pydantic import BaseModel
from src.tools.startup import logger

router = APIRouter(prefix="/index")


class Markdown(BaseModel):
    markdown: str


@router.post(
    path="/",
    responses={
        200: {"Description": "Correcly indexed the document"}})
def index_document(markdown: Markdown, request: Request):
    logger.info("Indexing document")
    logger.info(markdown)
