import whisper
from datetime import datetime
import os

class Transcriber:
    def __init__(self):
        self.whisper_model = whisper.load_model("base")

    def transcribe_audio(self, audio_file):
        """Transcribe audio file"""
        try:
            result = self.whisper_model.transcribe(audio_file)
            timestamp = datetime.now().strftime("%H:%M:%S")
            return f"[{timestamp}] {result['text']}"
        except Exception as e:
            print(f"Error during transcription: {e}")
            return None
