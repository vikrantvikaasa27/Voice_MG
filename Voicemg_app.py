import sounddevice as sd
import wavio
import whisper
from llama_index.llms import LlamaCPP
from llama_index.llms.base import ChatMessage
import streamlit as st
import time

# Function to record audio
def record_audio(output_filename, duration, sample_rate):
    st.write("Recording...")
    audio_data = sd.rec(int(duration * sample_rate),
                        samplerate=sample_rate, channels=1)
    timer_placeholder = st.empty()
    
    for remaining_time in reversed(range(int(duration))):
        timer_placeholder.text(f"Time remaining: {remaining_time} seconds")
        time.sleep(1)  # Wait for 1 second
    
    timer_placeholder.empty()  # Clear the timer placeholder
    sd.wait()  # Wait until recording is finished
    st.write("Recording finished.")
    wavio.write(output_filename, audio_data, sample_rate, sampwidth=2)

# Function to transcribe audio
def transcribe_audio(audio_file):
    model = whisper.load_model('small')
    text = model.transcribe(audio_file)
    return text['text']

# Function to check grammar and format
def check_grammar_and_format(text):
    path = r'C:\Users\vikra\llama.cpp\llama-2-13b-chat.ggmlv3.q4_0.bin'
    llm_gpt = LlamaCPP(model_path=path)
    message = ChatMessage(role='user', content=f'check grammar for the following: {text}')
    response = llm_gpt.chat([message])
    return response.message.content

def main():
    st.title("Speech-to-Text and Grammar Checking")
    st.subheader('How was your day?')

    recording_duration = st.sidebar.slider("Recording Duration (seconds)", 1, 60, 10)
    sample_rate = 44100

    # Button to start recording
    if st.button("Start Recording"):
        output_file = "recorded_audio.wav"
        record_audio(output_file, recording_duration, sample_rate)
        st.write("Audio saved as:", output_file)

        if not sd.query_devices(None, 'input')['default_samplerate'] == sample_rate:
            st.warning("Warning: The sample rate of the input device is not set to", sample_rate)

        # Use a spinner for transcription
        with st.spinner("Transcribing..."):
            transcribed_text = transcribe_audio(output_file)
            st.subheader("Transcribed Text:")
            st.write(transcribed_text)

        # Use a spinner for grammar check
        with st.spinner("Checking Grammar..."):
            grammar_check_result = check_grammar_and_format(transcribed_text)
            st.subheader("Grammar Check Result:")
            st.write(grammar_check_result)

    # Button to upload audio file
    uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'ogg'])

    if uploaded_file is not None:
        with open("uploaded_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Use a spinner for transcription
        with st.spinner("Transcribing uploaded audio..."):
            transcribed_text = transcribe_audio("uploaded_audio.wav")
            st.subheader("Transcribed Text:")
            st.write(transcribed_text)

    # Use a spinner for grammar check
        with st.spinner("Checking Grammar..."):
            grammar_check_result = check_grammar_and_format(transcribed_text)
            st.subheader("Grammar Check Result:")
            st.write(grammar_check_result)


if __name__ == "__main__":
    main()
