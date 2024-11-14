# routes/chat.py
from flask import Blueprint, request, render_template, jsonify
from services.chatutils import (
    get_pdf_text,
    get_text_chunks,
    get_vector_store,
    get_conversational_chain,
    search_internet,
    normalize_question,
)
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/')
def index():
    return render_template('index.html')

@chat_bp.route('/upload', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("files")
    text = get_pdf_text(uploaded_files)
    text_chunks = get_text_chunks(text)
    vectorstore = get_vector_store(text_chunks)
    return jsonify({"message": "Files processed successfully"})

@chat_bp.route('/ask', methods=['POST'])
def ask_question():
    user_question = request.form['question']
    normalized_question = normalize_question(user_question)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(normalized_question)
    
    chain = get_conversational_chain()
    response = chain.invoke({"input_documents": docs, "question": normalized_question})
    
    if not response["output_text"].strip() or "The provided documents do not contain this information" in response["output_text"]:
        internet_result = search_internet(normalized_question)
        response["output_text"] += f"\n\nInternet Search Result:\n{internet_result}"
    
    return jsonify({"response": response["output_text"]})
