import sounddevice as sd
import wavio
import whisper
import openai
from llama_index.llms import LlamaCPP
from llama_index.llms.base import ChatMessage


def record_audio(output_filename, duration, sample_rate):
    print("Recording...")
    audio_data = sd.rec(int(duration * sample_rate),
                        samplerate=sample_rate, channels=1)
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")

    # Save the recorded audio to a WAV file
    wavio.write(output_filename, audio_data, sample_rate, sampwidth=2)


def transcribe_audio(audio_file):
    model = whisper.load_model('base')
    text = model.transcribe(audio_file)
    return text['text']


def check_grammar_and_format(text):
    path = r'C:\Users\vikra\llama.cpp\llama-2-13b-chat.ggmlv3.q4_0.bin'
    llm_gpt = LlamaCPP(model_path=path)
    message = ChatMessage(role='user', content=f'check grammar and the correct format for the following: {text}')
    return llm_gpt.chat([message])


def main():
    print("Speech-to-Text and Grammar Checking")

    recording_duration = 5
    output_file = "recorded_audio.wav"
    sample_rate = 44100

    record_audio(output_file, recording_duration, sample_rate)
    print("Audio saved as:", output_file)

    if not sd.query_devices(None, 'input')['default_samplerate'] == sample_rate:
        print("Warning: The sample rate of the input device is not set to", sample_rate)

    transcribed_text = transcribe_audio(output_file)
    print("Transcribed Text:", transcribed_text)

    grammar_check_result = check_grammar_and_format(transcribed_text)
    print("Grammar Check Result:", grammar_check_result)


if __name__ == "__main__":
    main()
