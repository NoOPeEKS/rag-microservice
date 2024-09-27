from pydantic import BaseModel


class InputBody(BaseModel):
    file: str


class OutputBody(BaseModel):
    message: str
