# 🇹🇳 Tunisian_TTS


This is an Arabic text-to-speech (TTS) model fine-tuned on 300 hours of clean Modern Standard Arabic (MSA) audio. It delivers high-quality, natural speech synthesis with full diacritization support and efficient voice cloning from reference audio.

---

## Model Details

- Base Model: SparkAudio/Spark-TTS-0.5B
- Training Data: ~300 hours of clean Arabic audio
- Language: Modern Standard Arabic (MSA)
- Sample Rate: 24kHz
- Tags: speech, arabic, spark, tts, text-to-speech
- License: Fair Non-Commercial Research License

---

## Usage

### Quick Start

You can try the model directly via:
- Colab Notebook: https://colab.research.google.com/drive/11UN4qOUwCr3xS509ksZxbIrZf61pznSN?usp=sharing

```python
from transformers import AutoProcessor, AutoModel
import soundfile as sf
import torch

# Load model
model_id = "IbrahimSalah/Arabic-TTS-Spark"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModel.from_pretrained(model_id, trust_remote_code=True).eval().to(device)

# Prepare inputs
inputs = processor(
    text="YOUR_TEXT_WITH_TASHKEEL",
    prompt_speech_path="path/to/reference.wav",
    prompt_text="REFERENCE_TEXT_WITH_TASHKEEL",
    return_tensors="pt"
).to(device)

# Generate
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=8000, temperature=0.8)

# Decode
output = processor.decode(generated_ids=output_ids)
sf.write("output.wav", output["audio"], output["sampling_rate"])
```
## Key Features

- High-quality Arabic speech synthesis with natural prosody
- Efficient voice cloning from reference audio
- Advanced text chunking for long-form content
- Built-in audio post-processing (normalization, silence removal, crossfading)
- Adjustable generation parameters (temperature, top_k, top_p)

---

## Input Requirements

**Important:** Text must include full Arabic diacritization (tashkeel). The model performs poorly on non-diacritized input.

Example of correct input:
إِنَّ الْعِلْمَ نُورٌ يُقْذَفُ فِي الْقَلْبِ

### Generation Parameters

tts.generate_long_text(
    text=your_text,
    prompt_audio_path="reference.wav",
    prompt_transcript="reference_text",
    output_path="output.wav",
    max_chunk_length=300,        # Characters per chunk
    crossfade_duration=0.08,     # Crossfade duration in seconds
    normalize_audio_flag=True,
    remove_silence_flag=True,
    temperature=0.8,             # Generation randomness
    top_p=0.95,                  # Nucleus sampling
    top_k=50                     # Top-k sampling
)


---

## Further Fine-tuning

The model can be further fine-tuned for:
- Non-diacritized text
- Specific voice characteristics
- Domain-specific vocabulary
- Dialectal variations

Fine-tuning infrastructure: [Spark-TTS Fine-tune](https://github.com/tuan12378/Spark-TTS-finetune)

---

## Acknowledgments

- Base model: [Spark-TTS](https://github.com/tuan12378/Spark-TTS-finetune) by tuan12378

---

## Limitations

- Requires fully diacritized Arabic text as input
- Optimized for Modern Standard Arabic (MSA), not dialectal Arabic
- Performance may vary with very long texts without proper chunking
- Voice cloning quality depends on reference audio quality and length
- Generation speed scales with text length