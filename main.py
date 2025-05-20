from flask import Flask, request, jsonify
from pydub import AudioSegment
import openai
import os
import uuid
import shutil

app = Flask(__name__)

# Read OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

CHUNK_DURATION_MS = 15 * 60 * 1000  # 15 minutes

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    session_id = str(uuid.uuid4())
    os.makedirs(session_id, exist_ok=True)
    filepath = f"{session_id}/input.wav"
    file.save(filepath)

    try:
        audio = AudioSegment.from_file(filepath)
        chunks = [audio[i:i + CHUNK_DURATION_MS] for i in range(0, len(audio), CHUNK_DURATION_MS)]

        full_transcription = ""
        for idx, chunk in enumerate(chunks):
            chunk_path = f"{session_id}/chunk_{idx}.mp3"
            chunk.export(chunk_path, format="mp3")

            with open(chunk_path, "rb") as audio_file:
                response = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                full_transcription += f"[Parte {idx + 1}]\n{response.text}\n\n"

        return jsonify({"transcription": full_transcription})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        shutil.rmtree(session_id)

@app.route('/healthz')
def health():
    return "OK", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
