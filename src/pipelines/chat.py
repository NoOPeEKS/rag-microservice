from src.models.rag import RAG


def execute(settings: dict, global_settings: dict):
    rag = RAG(settings)
    print(rag.answer_question("How did juliet die?").split("\n")[-1])
    print(rag.answer_question("From which family is Romeo?").split("\n")[-1])
