from flask import Flask, request, jsonify
import os
import threading
import sys
from flask_cors import CORS

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from video_processor import process_video, history_data
from socket_instance import socketio

app = Flask(__name__)
CORS(app) 
socketio.init_app(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def health():
    return {"status": "Backend running"}

# ✅ FIXED HISTORY ROUTE
@app.route("/history", methods=["GET"])
def get_history():
    print("HISTORY:", history_data)
    return jsonify(history_data)

@app.route("/start", methods=["POST"])
def start():
    video = request.files["video"]
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    print("🎥 VIDEO RECEIVED:", video_path)

    t = threading.Thread(target=process_video, args=(video_path,))
    t.start()

    return jsonify({"status": "processing started"})

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000)