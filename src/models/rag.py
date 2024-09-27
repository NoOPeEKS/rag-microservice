import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import sentence_transformer
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "..", "..", "data", "chroma_db")
document_dir = os.path.join(current_dir, "..", "..", "data", "external", "romeo_and_juliet.txt")

class LocalEmbeddings:
    def __init__(self, model_name: str):
        self.embedding_model = SentenceTransformer(model_name)

    def embed_documents(self, documents):
        return self.embedding_model.encode(documents, show_progress_bar=True).tolist()

    def embed_query(self, query: str):
        return self.embedding_model.encode([query])[0].tolist()

class RAG:
    def __init__(self, settings: dict):
        if torch.cuda.is_available():
            self.__device = torch.device('cuda')
        else:
            self.__device = torch.device('cpu')

        self.tokenizer, self.llm = self._initialize_llm(settings['llm_model'])

        self.retriever = self._initialize_retriever(
            model_name=settings['embedding_model'],
            collection_name=settings['collection_name']
        )

    def _initialize_retriever(self, model_name: str, collection_name: str):
        #embedding_model = sentence_transformer.SentenceTransformerEmbeddings(model_name)
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        loader = TextLoader(document_dir)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=docs,
            collection_name=collection_name,
            embedding=embedding_model,
        )

        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        return retriever

    def _initialize_llm(self, model_name: str):
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.__device)

        return tokenizer, model

    def _generate_response(self, combined_input):
        inputs = self.tokenizer(combined_input, return_tensors="pt", truncation=True, max_length=2048).to(self.__device)
        
        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=512)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def answer_question(self, query):
        relevant_docs = self.retriever.get_relevant_documents(query)
        
        combined_input = (
            "You are a helpful assistant. "
            "Here are some documents that might help answer the question: "
            f"{query}\n\n"  # f-string interpolation for 'query'
            "Relevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])  # No f-string needed here
            + "\n\n"
            "Please provide an answer based only on the provided documents. "
            "If the answer is not found in the documents, respond with 'I'm not sure'."
        )

        result = self._generate_response(combined_input)
        return result
