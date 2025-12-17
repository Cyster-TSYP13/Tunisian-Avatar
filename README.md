# 🇹🇳 Tunisian‑Avatar

**Tunisian‑Avatar** is an open‑source AI assistant designed to support **Tunisian girls (ages 10–18)** with lifestyle guidance, daily tasks, and support for **PCOS (Polycystic Ovary Syndrome)** management all while communicating naturally in the Tunisian dialect.

This project combines multiple AI components (speech, text, and response generation) to create an interactive, friendly, and culturally relevant avatar experience.

The project’s three FastAPI services have been containerized, images pushed to an Azure Container Registry named **wietsypregistry**, and deployed on Azure Container Instances. Each service can be accessed via its public URL.

---

## 🚀 Project Overview

Tunisian‑Avatar empowers young users by:

- 🗣️ Speaking and understanding **Tunisian Arabic** naturally
- 💬 Providing lifestyle tips tailored to teenagers
- 🌱 Offering supportive guidance around **PCOS management**
- 🧠 Answering questions about daily life, health, and school
- 🤖 Using generative AI to handle conversations in a helpful, age-appropriate way

---

## 📂 Repository Structure

Tunisian‑Avatar/
├── Tunisian_Agentic_RAG      # Responsible for Reasoning & Response Generation
├── Tunisian_STT              # Speech‑to‑Text (converts spoken Tunisian Arabic into text)
├── Tunisian_TTS              # Text‑to‑Speech (generates spoken voice in Tunisian dialect)
├── README.md                 # This file
└── other files


---

## 🧠 Key Components

### 🗣️ Tunisian_STT — Speech Recognition

This module converts spoken Tunisian dialect into text for processing.  
It can be based on open‑source ASR models and fine‑tuned for Tunisian Arabic.


---

### 🎤 Tunisian_TTS — Speech Synthesis

Generates natural-sounding spoken responses in Tunisian Arabic.


---

### 🤖 Tunisian_Agentic_RAG — Conversational AI

Responsible for generating intelligent replies using retrieval and language models.  
Combines client input (from STT or text) with knowledge or context to produce supportive and relevant responses.


---

## 🧩 Features

✔️ Natural **Tunisian Arabic** interaction  
✔️ **Spoken and written** conversational support  
✔️ **Lifestyle & health** advice tailored for teenagers  
✔️ Designed to be **friendly, safe, and supportive**

---

## 💡 Getting Started

1. **Clone the repository**
git clone https://github.com/Cyster-TSYP13/Tunisian-Avatar.git
cd Tunisian-Avatar

2. **Install dependencies**
pip install -r requirements.txt

3. **Run modules independently**
- Run STT to get text from audio
- Use RAG for conversational responses
- Use TTS to speak responses

---

## 🧬 Health & Safety Considerations

This project aims to provide *informational support*  not *medical advice*. Always consult a healthcare professional for medical decisions, especially relating to **PCOS** or hormonal health.

---

## 📌 Contributing

Contributions are welcome! You can help by:

- Improving language models
- Enhancing dialect understanding
- Adding more health & lifestyle content
- Improving safety and moderation filters

Please open a pull request or issue with proposed changes.