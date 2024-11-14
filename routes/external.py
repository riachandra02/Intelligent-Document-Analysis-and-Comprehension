# routes/external.py
from flask import Blueprint, request, jsonify
from services.summ import get_pdf_text, get_text_chunks, summarize_text
from services.external import extract_keywords, search_papers, clean_paper_data,download_paper

external_bp = Blueprint('external', __name__)

@external_bp.route('/search_related', methods=['POST'])
def search_related_papers():
    """Endpoint to summarize PDF files, extract keywords and search for related papers."""
    try:
        uploaded_files = request.files.getlist("files")
        text = get_pdf_text(uploaded_files)
        text_chunks = get_text_chunks(text)
        
        if not text_chunks:
            return jsonify({"error": "No text chunks created from the uploaded files."}), 400
            
        # Generate the summary
        summary = summarize_text(text_chunks)
        
        # Extract keywords from the summary
        keywords = extract_keywords(summary)
        
        # Search for related papers using the keywords
        papers = search_papers(keywords)
        
        # Clean and validate paper data
        cleaned_papers = clean_paper_data(papers)
        
        return jsonify({
            "summary": summary,
            "keywords": keywords,
            "related_papers": cleaned_papers
        })
        
    except Exception as e:
        logger.error(f"Error in search_related_papers: {str(e)}")
        return jsonify({"error": str(e)}), 500

@external_bp.route('/extract_keywords', methods=['POST'])
def get_keywords():
    """Endpoint to extract keywords from the summary."""
    try:
        data = request.get_json()
        summary = data.get('summary', '')
        
        if not summary:
            return jsonify({"error": "No summary provided"}), 400
            
        keywords = extract_keywords(summary)
        return jsonify({"keywords": keywords})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
