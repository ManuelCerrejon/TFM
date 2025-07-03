import gradio as gr
import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel
from jiwer import wer

# Dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Modelo + processor
processor = WhisperProcessor.from_pretrained("./kaggle-v9")
base_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model = PeftModel.from_pretrained(base_model, "./kaggle-v9").to(device)

# Función para resaltar diferencias
def marcar_errores(pred, ref):
    pred_tokens = pred.strip().lower().split()
    ref_tokens = ref.strip().lower().split()
    resaltado = []

    for i, palabra in enumerate(pred_tokens):
        if i < len(ref_tokens) and palabra == ref_tokens[i]:
            resaltado.append(f"<span>{palabra}</span>")
        else:
            resaltado.append(f"<span style='color:red;font-weight:bold'>{palabra}</span>")

    return " ".join(resaltado)

# Función principal
def transcribe_and_compare(audio_path, referencia_txt):
    # Transcripción automática
    waveform, sample_rate = librosa.load(audio_path, sr=16000)
    input_features = processor(waveform, sampling_rate=16000, return_tensors="pt").input_features.to(device)
    predicted_ids = model.generate(input_features)
    transcripcion_pred = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Lectura del txt de referencia
    if referencia_txt:
        with open(referencia_txt, "r", encoding="utf-8") as f:
            transcripcion_ref = f.read().strip()
        wer_valor = wer(transcripcion_ref.lower(), transcripcion_pred.lower())
        wer_str = f"{wer_valor * 100:.2f}%"
        resaltado = marcar_errores(transcripcion_pred, transcripcion_ref)
    else:
        transcripcion_ref = "No se ha subido ningún archivo de referencia."
        wer_str = "N/A"
        resaltado = transcripcion_pred

    return resaltado, transcripcion_ref, wer_str

# Interfaz con HTML para resaltado
demo = gr.Interface(
    fn=transcribe_and_compare,
    inputs=[
        gr.Audio(type="filepath", label="Sube tu audio (.wav, 16 kHz preferido)"),
        gr.File(label="Sube archivo .txt con transcripción de referencia")
    ],
    outputs=[
        gr.HTML(label="Transcripción generada (con errores marcados en rojo)"),
        gr.Textbox(label="Transcripción de referencia"),
        gr.Textbox(label="WER (%)")
    ],
    title="DeepSea Speech",
    description="Simulador de transcripciones inteligente de radiocomunicaciones maritimas"
)

demo.launch(share=True)
