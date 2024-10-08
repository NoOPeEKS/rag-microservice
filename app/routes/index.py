from fastapi import APIRouter, Request
from app.models.markdown import Markdown
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.tools.startup import logger

router = APIRouter(prefix="/index")


@router.post(
    path="",
    responses={
        200: {"Description": "Correcly indexed the document"}})
def index_document(markdown: Markdown, request: Request):
    logger.info("Indexing document")
    logger.info(f"Document content: {markdown.markdown}")
    doc = Document(page_content=markdown.markdown)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents([doc])
    rag = request.app.rag
    rag.retriever.add_documents(docs)
    logger.info("Indexing documents finished")
