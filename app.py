# app.py
from flask import Flask
from routes.chat import chat_bp
from routes.summ import summ_bp
from routes.external import external_bp
from routes.voice_interactive import voice_interactive_bp

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(chat_bp)
app.register_blueprint(summ_bp)
app.register_blueprint(external_bp, url_prefix='/api')
app.register_blueprint(voice_interactive_bp)

if __name__ == "__main__":
    app.run(debug=True)
