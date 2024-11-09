#voice_interactive.py
from flask import Blueprint, jsonify, request
import speech_recognition as sr
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging
from services.chatutils import get_conversational_chain, normalize_question

voice_interactive_bp = Blueprint('voice-interactive', __name__, url_prefix='/voice-interactive')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.chain = get_conversational_chain()
        
    def get_answer_from_docs(self, question):
        """Get answer from the processed documents"""
        try:
            # Load the existing vector store
            vector_store = FAISS.load_local(
                "faiss_index",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            # Get relevant documents
            normalized_question = normalize_question(question)
            docs = vector_store.similarity_search(normalized_question)
            
            # Get response using the chain
            response = self.chain.invoke({
                "input_documents": docs,
                "question": normalized_question
            })
            
            return response.get("output_text", "").strip()
            
        except Exception as e:
            logger.error(f"Error getting answer: {e}")
            raise

    def transcribe_audio(self):
        """Capture and transcribe audio"""
        try:
            with sr.Microphone() as source:
                logger.info("Adjusting for ambient noise...")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                logger.info("Listening for speech...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=15)
                
                logger.info("Transcribing speech...")
                text = self.recognizer.recognize_google(audio)
                logger.info(f"Transcribed text: {text}")
                
                return text
                
        except sr.UnknownValueError:
            logger.error("Could not understand audio")
            raise ValueError("Could not understand audio")
        except sr.RequestError as e:
            logger.error(f"Could not request results: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            raise

@voice_interactive_bp.route('/start-conversation', methods=['POST'])
def start_conversation():
    """Handle voice input and return chat response"""
    try:
        voice_handler = VoiceHandler()
        
        # Get transcribed text
        transcribed_text = voice_handler.transcribe_audio()
        if not transcribed_text:
            return jsonify({"error": "No speech detected"}), 400
        
        # Get answer from documents
        response = voice_handler.get_answer_from_docs(transcribed_text)
        
        return jsonify({
            "transcribed_text": transcribed_text,
            "response": response
        })
        
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Error in conversation: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500
