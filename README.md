# 🔥 Am I The A**hole? — AI Judge Chatbot 🔥

An absurdly fun AI-powered chatbot that renders moral judgments on your life dilemmas. Powered by a configurable **Gemma** model via the Google GenAI API, with a **Python Azure Functions** backend and **React** frontend.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![React](https://img.shields.io/badge/React-18-61dafb?logo=react)
![Azure Functions](https://img.shields.io/badge/Azure_Functions-Python-0062ad?logo=azurefunctions)
![Gemma](https://img.shields.io/badge/Gemma-configurable-orange?logo=google)

## ✨ Features

- 🤖 **AI-Powered Judgments** — Real-time streaming responses from a configurable Gemma model
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
- A Google AI Studio API key (for Gemma access)
- Azure Functions Core Tools

### Azure Functions API Setup

```bash
cd api
python -m venv .venv

# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `local.settings.json.example` to the ignored `local.settings.json` file and fill in the Google GenAI and PostgreSQL settings for your environment.

Start the server:

```bash
func start
```

The Azure Functions API runs on `http://localhost:7071` by default.

The Flask application under `backend/` is retained as a legacy, non-production implementation. The deployed application uses the Azure Functions under `api/`.

### Frontend Setup

```bash
cd frontend
npm install
npm start
```

For local development, set `REACT_APP_API_URL=http://localhost:7071` in `frontend/.env.local`. The React dev server runs on `http://localhost:3000`; production uses same-origin `/api` routes.

## 📁 Project Structure

```
AITAChatbot/
├── api/                        # Production Azure Functions API
│   ├── shared_code/        # GenAI, validation, and request controls
│   ├── db.py               # PostgreSQL counter access
│   └── requirements.txt    # Production Python dependencies
├── backend/                    # Legacy non-production Flask implementation
├── frontend/
│   ├── public/
│   │   └── index.html      # HTML template with Google Fonts
│   ├── src/
│   │   ├── App.js          # Main React component
│   │   ├── App.css         # All styling (glassmorphism, animations)
│   │   └── index.js        # React entry point
│   └── package.json        # Node dependencies
├── tests/                      # Python unit tests
├── .github/workflows/          # Azure Static Web Apps deployment
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

- **Backend:** Python, Azure Functions, google-genai
- **Frontend:** React 18, react-markdown, CSS3 (custom, no frameworks)
- **Data:** PostgreSQL via psycopg
- **AI Model:** Configurable Gemma model via the Google GenAI API
- **Deployment:** Azure Static Web Apps with integrated Azure Functions
- **Streaming:** Server-Sent Events (SSE)

## 🎭 Judgment Types

The UI maps the model's structured court verdict to the corresponding AITA-style badge:

| Code | Meaning |
|------|---------|
| YTA 🫵 | `The Court Declares: Guilty!` |
| NTA ✅ | `The Court Declares: Not Guilty!` |

## ⚠️ Disclaimer

This AI judge has **zero legal authority** and a **questionable moral compass**. It's powered by a language model with no life experience whatsoever. For entertainment purposes only. Please don't sue us. 🎭

## 📄 License

MIT
