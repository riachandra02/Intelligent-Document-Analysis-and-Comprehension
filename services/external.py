#services/external.py
import io
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from typing import List
import PyPDF2

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

def extract_keywords_from_pdf(pdf_file) -> List[str]:
    """
    Extract keywords from a PDF file using frequency analysis and POS tagging
    """
    try:
        # Read PDF content
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        
        # Extract text from each page
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + " "

        # Log the extracted text
        print("Extracted text:", text)

        if not text.strip():
            print("No text could be extracted from PDF")
            return []

        # Tokenize and clean text
        tokens = word_tokenize(text.lower())
        print("Tokens:", tokens)  # Log tokens

        stop_words = set(stopwords.words('english'))
        
        # Remove stopwords, numbers, and short words
        words = [word for word in tokens 
                if word.isalnum() and 
                word not in stop_words and 
                len(word) > 3]

        print("Filtered words:", words)  # Log filtered words

        # Get POS tags
        pos_tags = nltk.pos_tag(words)
        
        # Keep only nouns and adjectives
        keywords = [word for word, pos in pos_tags 
                   if pos.startswith(('NN', 'JJ'))]

        print("Keywords found:", keywords)  # Log keywords

        # Count frequency
        word_freq = Counter(keywords)
        
        # Get top 5 most common keywords
        top_keywords = [word for word, _ in word_freq.most_common(5)]
        
        if not top_keywords:
            print("No keywords found after processing")
            return []

        return top_keywords

    except Exception as e:
        print(f"Error extracting keywords: {str(e)}")
        return []

