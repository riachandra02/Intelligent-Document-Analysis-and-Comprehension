# routes/summ.py
from flask import Blueprint, request, jsonify
from services.summ import get_pdf_text, get_text_chunks, summarize_text

summ_bp = Blueprint('summ', __name__)

@summ_bp.route('/summarize', methods=['POST'])
def summarize_files():
    uploaded_files = request.files.getlist("files")
    text = get_pdf_text(uploaded_files)
    text_chunks = get_text_chunks(text)

    # Debugging: Print text and text chunks
    print("Extracted Text:", text)
    print("Text Chunks:", text_chunks)

    if not text_chunks:
        return jsonify({"error": "No text chunks created from the uploaded files."}), 400

    summary = summarize_text(text_chunks)
    return jsonify({"summary": summary})
