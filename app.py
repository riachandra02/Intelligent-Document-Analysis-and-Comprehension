# app.py
from flask import Flask
from routes.chat import chat_bp  # Import your routes
from routes.summ import summ_bp

app = Flask(__name__)

# Register the blueprint
app.register_blueprint(chat_bp)
app.register_blueprint(summ_bp)

if __name__ == "__main__":
    app.run(debug=True)
