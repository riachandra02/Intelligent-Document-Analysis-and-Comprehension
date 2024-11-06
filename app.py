# app.py
from flask import Flask
from routes.chat import chat_bp  # Import your routes
from routes.summ import summ_bp
from routes.external import fetch_external_bp
from routes.voice import voice_bp

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(chat_bp)
app.register_blueprint(summ_bp)
app.register_blueprint(fetch_external_bp)
app.register_blueprint(voice_bp)

if __name__ == "__main__":
    app.run(debug=True)
