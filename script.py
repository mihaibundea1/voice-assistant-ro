import sys
import io
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from transformers import pipeline, VitsModel, AutoTokenizer, AutoModelForCausalLM

# Forțează encoding UTF-8 pentru consolă
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Configurare parametri
FS = 16000
DURATA_INREGISTRARE = 5

print("Aplicația a pornit! ✅")

# Restul codului rămâne neschimbat...

# 1. Înregistrare audio
def inregistreaza_audio():
    print("🎤 Vorbeste acum...")
    audio = sd.rec(int(DURATA_INREGISTRARE * FS), samplerate=FS, channels=1, dtype='float32')
    sd.wait()
    write("input.wav", FS, audio)
    return audio

# 2. Inițializare modele
try:
    # Speech-to-Text (STT)
    stt_pipeline = pipeline(
        "automatic-speech-recognition",
        model="gigant/whisper-medium-romanian",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Text-to-Speech (TTS)
    tts_model = VitsModel.from_pretrained("facebook/mms-tts-ron")
    tts_tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-ron")
    
    # LLM - înlocuit cu model adecvat pentru generare text
    llm_model = AutoModelForCausalLM.from_pretrained("readerbench/RoGPT2-medium")
    llm_tokenizer = AutoTokenizer.from_pretrained("readerbench/RoGPT2-medium")

except Exception as e:
    print(f"🚨 Eroare la inițializarea modelelor: {e}")
    exit()

# 3. Pipeline complet
def proceseaza_comanda():
    try:
        # Înregistrare audio
        inregistreaza_audio()
        
        # Speech-to-Text
        result = stt_pipeline("input.wav")
        text_input = result.get("text", "Nu s-a putut recunoaște textul")
        print(f"🔍 Text recunoscut: {text_input}")
        
        # Procesare LLM
        inputs = llm_tokenizer(
            text_input,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding="max_length"
        )
        outputs = llm_model.generate(
            inputs.input_ids,
            max_length=100,
            num_beams=5,
            early_stopping=True
        )
        text_output = llm_tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"🤖 Răspuns LLM: {text_output}")
        
        # Text-to-Speech
        inputs = tts_tokenizer(text_output, return_tensors="pt")
        with torch.no_grad():
            waveform = tts_model(inputs["input_ids"]).waveform
        
        # Redare audio
        sd.play(waveform.numpy().squeeze(), samplerate=16000)
        sd.wait()

    except Exception as e:
        print(f"🚨 Eroare în procesare: {e}")

if __name__ == "__main__":
    proceseaza_comanda()