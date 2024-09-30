import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import chat
from src.tools.startup import params

app = FastAPI()
app.include_router(chat.router)

api_settings = params['api']

app.add_middleware(CORSMiddleware, **api_settings['middleware'])


@app.get("/")
def read_root():
    return {"message": "SISMOTUR API Service"}


if __name__ == "__main__":
    uvicorn.run("app.main:app", **api_settings['config'])
