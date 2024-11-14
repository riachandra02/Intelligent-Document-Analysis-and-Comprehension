# services/utils.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from googlesearch import search

def get_pdf_text(pdf_docs):
    """Extract text from a list of PDF documents."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into chunks using a RecursiveCharacterTextSplitter."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save a FAISS vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """Create a conversational chain with a custom prompt template."""
    prompt_template = """
    You are a helpful and informative bot that answers questions using text from the reference Context included below. \
    Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
    However, you are talking to a non-technical audience and technical audience as well, so be sure to break down complicated concepts and \
    strike a friendly and conversational tone. \
    If the Context is irrelevant to the answer, you may tell the user what the provided context is and it is not available in the provided context and give the answer from the internet."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.2)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

def search_internet(query):
    """Search the internet for the answer."""
    search_results = list(search(query, num_results=1))  # Convert generator to list and fetch only the top result
    if search_results:
        return search_results[0]
    return "No relevant information found on the internet."

def normalize_question(question):
    """Normalize the user question to ensure consistency."""
    question = question.lower().strip()
    return question

