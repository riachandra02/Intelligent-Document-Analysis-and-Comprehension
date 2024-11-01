# services/summ.py
import os
import logging
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_pdf_text(pdf_files):
    """Extract text from PDF files."""
    try:
        text = ""
        for pdf in pdf_files:
            logger.debug(f"Processing PDF: {pdf.filename}")
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        logger.debug(f"Extracted text length: {len(text)}")
        return text
    except Exception as e:
        logger.error(f"Error in get_pdf_text: {str(e)}")
        raise

def get_text_chunks(text):
    """Split text into chunks."""
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        logger.debug(f"Created {len(chunks)} text chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error in get_text_chunks: {str(e)}")
        raise

def summarize_text(text_chunks):
    """Summarize the text chunks using Google Generative AI."""
    try:
        # Check if API key is set
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is not set")
        
        logger.debug("Initializing Google Generative AI model")
        llm = ChatGoogleGenerativeAI(model="gemini-pro")
        
        # Combine chunks with appropriate spacing
        full_text = " ".join(text_chunks)
        logger.debug(f"Combined text length: {len(full_text)}")
        
        prompt_template = """
        Please provide a comprehensive summary of the following text. Focus on the main points and key insights:

        {text}

        Summary:
        """
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
        chain = LLMChain(llm=llm, prompt=prompt)
        
        logger.debug("Generating summary")
        summary = chain.run(text=full_text)
        logger.debug(f"Summary generated, length: {len(summary)}")
        
        return summary.strip()
    except Exception as e:
        logger.error(f"Error in summarize_text: {str(e)}")
        raise
