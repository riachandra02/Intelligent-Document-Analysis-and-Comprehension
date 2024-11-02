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
        
        # Keep only nouns and adjectives for keyword relevance
        keywords = [word for word, pos in pos_tags 
                    if pos.startswith(('NN', 'JJ'))]

        # Count frequency of each keyword
        word_freq = Counter(keywords)
        
        # Extract the top 5 most common keywords
        top_keywords = [word for word, _ in word_freq.most_common(5)]
        
        return top_keywords

    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

@fetch_external_bp.route('/fetch-external-data', methods=['POST'])
def fetch_external_data():
    """Process uploaded PDFs and fetch related research data from CORE API based on extracted keywords."""
    try:
        # Check if files were uploaded
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

        results = []

        # Process each PDF
        for file in files:
            try:
                if not file.filename.endswith('.pdf'):
                    results.append({
                        "filename": file.filename,
                        "error": "Not a PDF file"
                    })
                    continue

                # Extract keywords from the PDF
                keywords = extract_keywords_from_pdf(file)
                if not keywords:
                    results.append({
                        "filename": file.filename,
                        "error": "No keywords extracted"
                    })
                    continue

                # Query CORE API with extracted keywords
                query = " OR ".join(keywords)
                response = requests.get(
                    CORE_API_URL,
                    headers={"Authorization": f"Bearer {CORE_API_KEY}"},
                    params={"q": query, "limit": 10}
                )

                if response.status_code != 200:
                    results.append({
                        "filename": file.filename,
                        "error": f"CORE API request failed with status code {response.status_code}"
                    })
                    continue

                # Collect and structure the results from CORE API
                articles = response.json().get('results', [])
                article_links = [{"title": article["title"], "link": article["urls"]["core"]} for article in articles]

                results.append({
                    "filename": file.filename,
                    "articles": article_links if article_links else "No relevant articles found."
                })

            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": f"Error processing file: {str(e)}"
                })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({
            "error": "An unexpected error occurred",
            "debug": str(e)
        }), 500
