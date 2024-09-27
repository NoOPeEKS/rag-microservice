import bs4
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
import torch

# Load the webpage and parse it with bs4
loader = WebBaseLoader(
    web_path="https://lilianweng.github.io/posts/2023-06-23-agent/",
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)
docs = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Initialize SentenceTransformer model for embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
class LocalEmbeddings:
    def embed_documents(self, documents):
        return embedding_model.encode(documents, show_progress_bar=True).tolist()
    def embed_query(self, query):
        return embedding_model.encode([query])[0].tolist()

# Create vectorstore with the local embeddings
vectorstore = Chroma.from_documents(documents=splits, embedding=LocalEmbeddings())
retriever = vectorstore.as_retriever()

# Load the prompt template
prompt = hub.pull("rlm/rag-prompt")

# Load StableLM model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("stabilityai/stablelm-zephyr-3b")
llm = AutoModelForCausalLM.from_pretrained("stabilityai/stablelm-zephyr-3b")

# Function to format documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG chain
def rag_chain(context, question):
    # Format the input using the prompt template
    input_prompt = prompt.format(context=context, question=question)
    
    # Tokenize the input text
    inputs = tokenizer(input_prompt, return_tensors="pt", truncation=True)

    # Generate the output from the model
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=100)
    
    # Decode the output tokens to get the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

