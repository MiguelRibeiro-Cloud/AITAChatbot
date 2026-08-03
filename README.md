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
# Optional: defaults to 1024, comfortably above the prompt's under-150-word reply target.
GEMINI_MAX_OUTPUT_TOKENS=1024
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
| `/api/cases-heard` | GET | Read the deployment-wide cases-heard total |
| `/api/chat` | POST | Send message, get full response |
| `/api/chat/stream` | POST | Send message, get SSE streamed response |

### Deployment Counter Settings

The Azure Functions API reads the deployment-wide cases-heard counter from PostgreSQL using these app settings:

```env
POSTGRES_HOST=your-server.postgres.database.azure.com
POSTGRES_PORT=5432
POSTGRES_DATABASE=aitabot
POSTGRES_USER=aitabot_app
POSTGRES_PASSWORD=replace-locally
POSTGRES_SSLMODE=require
```

### Deployment Security Settings

The deployed Azure Static Web Apps frontend uses `frontend/public/staticwebapp.config.json` for security headers and routing hardening. Production source maps are disabled by the frontend build script and `*.map` files under `/static/` are blocked by hosting config.

The chat API endpoints are anonymous public endpoints, but enforce server-side validation, in-process request throttling, provider timeouts, maximum message length, maximum history length, and maximum provider output tokens. For high-traffic production use, replace the in-process throttle with a durable shared rate limiter or API gateway policy so limits apply consistently across all function instances.

The public health endpoint only reports service liveness. Keep model names, API-key presence, and environment diagnostics in authenticated operational tooling or logs, not in public responses.

Submitted stories are sent from the browser to the Azure Functions API and then to Google GenAI for response generation. The application stores only the deployment-wide case count in PostgreSQL and the browser stores only disclaimer acceptance. Do not submit names, addresses, workplaces, contact details, or other identifying information.

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

### Error Codes

The API can return the following HTTP status codes for chat endpoints (`/api/chat` and `/api/chat/stream`).

| Status | Where | Brief Meaning | Possible Causes |
|--------|-------|---------------|-----------------|
| 400 | API validation | Bad request payload | Missing `message`, empty `message`, or message exceeds 10,000 characters. |
| 401 | GenAI auth | AI credentials/permissions issue | Missing/invalid `GEMINI_API_KEY`, revoked key, or key lacks permission for the configured model/project. |
| 429 | GenAI quota/rate | Usage limit reached | Provider quota exhausted, rate limit exceeded, or token usage cap reached. |
| 502 | GenAI model lookup | Configured model unavailable | Wrong `GEMINI_MODEL_NAME`, model removed/deprecated, typo in model identifier, or model not enabled for the account. |
| 503 | GenAI provider availability | Provider temporarily overloaded | Upstream provider high demand or temporary service unavailability. |
| 500 | API internal | Unexpected server-side failure | Unclassified provider errors, runtime exceptions, malformed upstream responses, or unknown edge cases. |

Notes:

- `200` can still include the fallback text (`"I... I got nothing. My brain is empty. Like a coconut."`) when the provider call succeeds but returns empty text.
- `/api/health` returns `200` when the API is alive; it does not validate GenAI key/model correctness.

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
