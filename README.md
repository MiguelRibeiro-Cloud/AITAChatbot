# 🔥 Am I The A**hole? — AI Judge Chatbot 🔥

An absurdly fun AI-powered chatbot that renders moral judgments on your life dilemmas. Powered by **Gemma 3 12B** via the Google GenAI API, with a **Flask** backend and **React** frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![Flask](https://img.shields.io/badge/Flask-3.1-black?logo=flask)
![Gemma](https://img.shields.io/badge/Gemma_3-12B-orange?logo=google)

## ✨ Features

- 🤖 **AI-Powered Judgments** — Real-time streaming responses from Gemma 3 12B
- ⚡ **Live Streaming** — Watch the verdict unfold in real-time via SSE
- 🎨 **Ridiculous Design** — Glassmorphism, animated blobs, gradient everything
- 📋 **Copy Responses** — One-click copy on any AI response
- 📥 **Export Transcripts** — Save your verdict as a `.txt` file
- 🔄 **Error Retry** — Failed? Hit retry without retyping
- ⏹️ **Stop Generation** — Cancel mid-response with Stop button or Escape key
- 💬 **Conversation History** — Multi-turn context (last 20 messages)
- 📱 **Responsive** — Works on mobile, tablet, and desktop
- 🛡️ **Rate Limiting** — Basic IP-based rate limiting (20 req/min)

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- A Google AI Studio API key (for Gemma 3 access)

### Backend Setup

```bash
cd backend
python -m venv venv

# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
```

Create a `.env` file in `backend/`:

```env
GEMINI_API_KEY=your_api_key_here
```

Start the server:

```bash
python app.py
```

The Flask server runs on `http://localhost:5000`.

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

The React dev server runs on `http://localhost:3000` and proxies API calls to the backend.

## 📁 Project Structure

```
AmItheAssohole/
├── backend/
│   ├── app.py              # Flask API server
│   ├── requirements.txt    # Python dependencies
│   └── .env                # API key (not committed)
├── frontend/
│   ├── public/
│   │   └── index.html      # HTML template with Google Fonts
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── App.css         # All styling (glassmorphism, animations)
│   │   └── index.js        # React entry point
│   └── package.json        # Node dependencies
├── .gitignore
└── README.md
```

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check — confirms the judge is in |
| `/api/chat` | POST | Send message, get full response |
| `/api/chat/stream` | POST | Send message, get SSE streamed response |

### Request Body (POST endpoints)

```json
{
  "message": "AITA for eating my roommate's leftovers?",
  "history": [
    { "role": "user", "content": "previous message" },
    { "role": "assistant", "content": "previous response" }
  ]
}
```

## 🛠️ Tech Stack

- **Backend:** Python, Flask, Flask-CORS, google-genai, python-dotenv
- **Frontend:** React 18, react-markdown, CSS3 (custom, no frameworks)
- **AI Model:** Gemma 3 12B IT (via Google GenAI API)
- **Streaming:** Server-Sent Events (SSE)

## 🎭 Judgment Types

The AI delivers verdicts tagged with classic AITA judgments:

| Code | Meaning |
|------|---------|
| YTA 🫵 | You're The A**hole |
| NTA ✅ | Not The A**hole |
| ESH 💀 | Everyone Sucks Here |
| INFO 🤔 | Not Enough Info |
| NAH 🤷 | No A**holes Here |

## ⚠️ Disclaimer

This AI judge has **zero legal authority** and a **questionable moral compass**. It's powered by a language model with no life experience whatsoever. For entertainment purposes only. Please don't sue us. 🎭

## 📄 License

MIT
