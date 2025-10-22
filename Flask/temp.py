from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask is running!"

# ✅ This must be at the bottom
if __name__ == '__main__':
    app.run(debug=True)
