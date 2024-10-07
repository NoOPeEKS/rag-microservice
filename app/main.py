import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from app.routes import chat, index
from src.models.rag import RAG
from src.tools.startup import params


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.rag = RAG(params['global']['rag_chatbot'])
    yield


def get_rag():
    return app.rag


app = FastAPI(lifespan=lifespan)
app.include_router(chat.router)
app.include_router(index.router)

api_settings = params['api']

app.add_middleware(CORSMiddleware, **api_settings['middleware'])


@app.get("/")
def read_root():
    return {"message": "SISMOTUR API Service"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", **api_settings['config'])
