import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_dir = os.path.join(current_dir, "..", "..", "data", "chroma_db")
document_dir = os.path.join(
    current_dir, "..", "..", "data",
    "external", "romeo_and_juliet.txt"
)


class RAG:
    """
    Class that encapsulates all of the Retrieval-Augmented Generation
    functionality.
    """
    def __init__(self, settings: dict) -> None:
        """
        Constructor. THis method does the following:
        - Set internal attributes
        - Initialize the tokenizer
        - Initialize the llm
        - Initialize the retriever
        """
        if torch.cuda.is_available():
            self.__device = torch.device('cuda')
        else:
            self.__device = torch.device('cpu')

        self.tokenizer, self.llm = self._initialize_llm(settings['llm_model'])

        self.retriever = self._initialize_retriever(
            model_name=settings['embedding_model'],
            collection_name=settings['collection_name']
        )

    def _initialize_retriever(
            self,
            model_name: str,
            collection_name: str) -> VectorStoreRetriever:
        """
        Initializes the document retriever. First sets the embedding model,
        then loads documents, splits documents and text and then initializes
        ChromaDB as a retriever.

        Args:
            model_name: The name of the model in HuggingFace.
            collection_name: The name of the collection where to store docs.

        Returns:
            retriever: The VectorStoreRetriever object.
        """
        embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        loader = TextLoader(document_dir)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=0
        )
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
        """
        Initializes the llm for the RAG. Also initializes the tokenizer.

        Args:
            model_name: The name of the model in HuggingFace.

        Returns:
            tokenizer: The pretrained tokenizer for the model.
            model: The pretrained LLM model.
        """
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(self.__device)

        return tokenizer, model

    def _generate_response(self, combined_input: str) -> str:
        """
        Generates the response based on a provided combined input.
        Tokenizes the input with the tokenizer and generates output
        with LLM.

        Args:
            combined_input: The combined input prompt for the LLM response.

        Returns:
            response: The response text to the provided prompt.
        """
        inputs = self.tokenizer(
            combined_input, return_tensors="pt",
            truncation=True, max_length=2048
        ).to(self.__device)

        with torch.no_grad():
            outputs = self.llm.generate(**inputs, max_new_tokens=512)

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response

    def answer_question(self, query: str) -> str:
        """
        Generates the answer for the provided question. Retrieves relevant
        documents for the query, joins combines the documents and query
        in a new prompt and asks the LLM for the answer.

        Args:
            query: The question text.

        Returns:
            result: The answer to the given query
        """
        relevant_docs = self.retriever.get_relevant_documents(query)

        combined_input = (
            "You are a helpful assistant. "
            "Here are some documents that might help answer the question: "
            f"{query}\n\n"
            "Relevant Documents:\n"
            + "\n\n".join([doc.page_content for doc in relevant_docs])
            + "\n\n"
            "Please provide an answer based only on the provided documents. "
            "If the answer is not found in the documents, respond with "
            "'I'm not sure'."
        )

        result = self._generate_response(combined_input)
        return result
