import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
#from htmlTemplates import css, bot_template, user_template
from googlesearch import search  # Import the google search module

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

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
    strike a friendly and converstional tone. \
    If the Context is irrelevant to the answer, you may tell the user what the provided context is and it is not available in the provided context and give the answer from internet."

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

def handle_user_input(user_question):
    """Handle user input and generate a response."""
    normalized_question = normalize_question(user_question)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(normalized_question)
    
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": normalized_question})
    
    if not response["output_text"].strip() or "The provided documents do not contain this information" in response["output_text"]:
        internet_result = search_internet(normalized_question)
        response["output_text"] += f"\n\nInternet Search Result:\n{internet_result}"
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": response["output_text"]})

    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.write(f"User: {message['content']}")
        else:
            st.write(f"Bot: {message['content']}")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    # If you have HTML templates, you can include them here
    #st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversational_chain()

if __name__ == "__main__":
    main()
