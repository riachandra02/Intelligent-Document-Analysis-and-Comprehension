# routes/voice.py
from flask import Blueprint, jsonify, request
import speech_recognition as sr
import logging

voice_bp = Blueprint('voice', __name__, url_prefix='/voice')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@voice_bp.route('/record-voice', methods=['POST'])
def record_voice():
    try:
        recognizer = sr.Recognizer()
        
        # Configure recognition parameters
        recognizer.energy_threshold = 4000
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        
        with sr.Microphone() as source:
            logger.info("Initializing microphone...")
            # Adjust for ambient noise
            recognizer.adjust_for_ambient_noise(source, duration=1)
            logger.info("Listening for speech...")
            
            # Set timeout and phrase_time_limit to avoid hanging
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            
            logger.info("Audio captured, processing...")
            
            # Try multiple recognition services
            try:
                text = recognizer.recognize_google(audio)
                return jsonify({"text": text})
            except sr.RequestError:
                logger.error("Google recognition failed, falling back to Sphinx")
                try:
                    text = recognizer.recognize_sphinx(audio)
                    return jsonify({"text": text})
                except Exception as sphinx_error:
                    logger.error(f"Sphinx recognition failed: {sphinx_error}")
                    raise
                    
    except sr.UnknownValueError:
        logger.error("Speech was not understood")
        return jsonify({"error": "Could not understand speech. Please speak clearly and try again."}), 400
    except sr.RequestError as e:
        logger.error(f"Speech recognition service error: {e}")
        return jsonify({"error": "Speech recognition service is currently unavailable."}), 503
    except Exception as e:
        logger.error(f"Unexpected error in voice recognition: {e}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
