# LLM-Test-case-generator

# 🧪 AI-Powered Test Case Generator

An internal-use prototype built to generate test cases using an LLM (Google Gemma) based on user prompts. Designed with an async job pipeline to avoid blocking the frontend and deliver real-time job status updates through HTTP polling.

> ⚠️ This is a **reconstructed version** of an internal tool I initiated  for potential QA automation adoption.

---

## 🚀 Key Features

- 🧠 Accepts user prompt or problem statement input
- 🤖 Uses **Google Gemma** (LLM) to generate relevant test cases
- 🕒 Async job queue on backend to prevent UI blocking
- 🔁 HTTP polling from frontend to check for job status and retrieve results
- 🎯 Designed for fast, testable UX feedback loop

---

## 🧩 System Architecture


React.js (Frontend)
  ├── Submits prompt via POST
  └── Polls backend for job status (HTTP)

Flask (Backend)
  ├── Receives request and queues LLM job
  ├── Runs async task (thread/queue) to call Google Gemma
  └── On completion, stores result and exposes via GET


---


## ⚙️ Tech Stack

- **Frontend**: React.js (Vite)
- **Backend**: Python Flask
- **LLM**: Google Gemma (text-generation model)
- **Task Management**: Python threading & job queue
- **Communication**: HTTP polling (long-poll fallback)

---

## 🛠️ Local Setup

```bash
# Clone the repo
git clone https://github.com/ArjunR00T/llm-testgen
cd llm-testgen

# Start backend
cd backend
pip install -r requirements.txt
python run.py

# Start frontend
cd ../frontend
npm install
npm run dev
```

📌 Notes
Current version uses threading for async handling — can be upgraded to Celery or RQ for production-scale job queues.

LLM integration is abstracted behind a generate_test_case() module for model flexibility.

Designed during internal R&D 

### 🙌 Author
Built by Arjun Gireesh
Backend Developer | Microservices • Queues • Async Architectures
Now open to backend engineering roles at product-focused teams.

