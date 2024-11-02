# routes/external.py
import requests
from flask import Blueprint, request, jsonify
import PyPDF2
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from typing import List, Dict
import io  # Add this import

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

fetch_external_bp = Blueprint('fetch_external', __name__)

fetch_external_bp = Blueprint('fetch_external', __name__)

CORE_API_KEY = "WGVRl3nBb6Hxevh7TzU5DiIS8XQKwfOq"
CORE_API_URL = "https://api.core.ac.uk/v3/search/works"

def extract_keywords_from_pdf(pdf_file) -> List[str]:
    """
    Extract keywords from a PDF file using frequency analysis and POS tagging
    """
    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords, numbers, and short words
        words = [word for word in tokens 
                if word.isalnum() and 
                word not in stop_words and 
                len(word) > 3]

        # Get POS tags
        pos_tags = nltk.pos_tag(words)
        
        # Keep only nouns and adjectives
        keywords = [word for word, pos in pos_tags 
                   if pos.startswith(('NN', 'JJ'))]

        # Count frequency
        word_freq = Counter(keywords)
        
        # Get top 5 most common keywords
        top_keywords = [word for word, _ in word_freq.most_common(5)]
        
        return top_keywords

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

# routes/external.py
# routes/external.py
@fetch_external_bp.route('/fetch-external-data', methods=['POST'])
def fetch_external_data():
    """Simplified version for testing PDF processing"""
    try:
        # Check if any files were uploaded
        if 'files' not in request.files:
            return jsonify({
                "error": "No files in request",
                "debug": {"step": "file_check"}
            }), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({
                "error": "No files provided",
                "debug": {"step": "files_list"}
            }), 400

        # Process each PDF file
        results = []
        for file in files:
            try:
                # Basic file checks
                if not file.filename:
                    continue
                
                if not file.filename.endswith('.pdf'):
                    results.append({
                        "filename": file.filename,
                        "error": "Not a PDF file"
                    })
                    continue

                # Read the file content
                file_content = file.read()
                if not file_content:
                    results.append({
                        "filename": file.filename,
                        "error": "Empty file"
                    })
                    continue

                # Try to read the PDF
                pdf_file = io.BytesIO(file_content)
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                
                # Try to extract text from the first page
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                
                results.append({
                    "filename": file.filename,
                    "pages": len(pdf_reader.pages),
                    "first_page_chars": len(text),
                    "sample_text": text[:100] if text else "No text extracted"
                })

            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })

        if not results:
            return jsonify({
                "error": "No files were processed",
                "debug": {"step": "processing"}
            }), 400

        return jsonify({
            "status": "success",
            "results": results
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "debug": {
                "step": "general",
                "error_type": str(type(e).__name__)
            }
        }), 500
