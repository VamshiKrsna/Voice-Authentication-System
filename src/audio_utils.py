import sounddevice as sd
import soundfile as sf

class AudioRecorder:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def record_audio(self, duration):
        """Record audio from microphone"""
        print(f"Recording for {duration} seconds...")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return audio.flatten()

    def save_audio(self, audio, filename):
        """Save audio to file"""
        sf.write(filename, audio, self.sample_rate)
