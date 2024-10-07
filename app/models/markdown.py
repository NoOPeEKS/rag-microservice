from pydantic import BaseModel


class Markdown(BaseModel):
    markdown: str
