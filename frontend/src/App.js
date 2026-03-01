import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import './App.css';

const JUDGMENTS = [
  "YTA 🫵", "NTA ✅", "ESH 💀", "INFO 🤔", "NAH 🤷",
];

const PLACEHOLDERS = [
  "So I told my roommate their cooking smells like defeat...",
  "AITA for naming my WiFi 'YourMusicIsTooLoud'?",
  "I ate my coworker's labeled lunch. In my defense, I was hungry.",
  "My cat knocked over my friend's drink and I laughed...",
  "I told a toddler Santa isn't real. The toddler was my boss's kid.",
  "AITA for responding to every work email with just 'K'?",
  "I replaced all the office coffee with decaf. It's been 3 weeks.",
  "AITA for telling my in-laws their casserole tastes like regret?",
];

const WELCOME_MESSAGES = [
  "Welcome to the Court of Chaotic Judgment! 🏛️⚖️",
  "Tell me your tale of possible a**holery and I shall render my TOTALLY unbiased verdict.",
  "I'm powered by Gemma 3 — an AI with zero life experience but VERY strong opinions.",
];

const LOADING_QUIPS = [
  "Consulting the ancient scrolls of social etiquette...",
  "Surveying 10,000 Redditors in a parallel universe...",
  "Calculating your a**hole coefficient...",
  "Weighing your moral fiber on quantum scales...",
  "Asking my therapist about YOUR problems...",
  "Generating righteous indignation...",
  "Cross-referencing the Universal Jerk Database...",
  "Summoning the ghost of Emily Post...",
];

const API_BASE = process.env.REACT_APP_API_URL || "";

function TypingIndicator() {
  const [quip, setQuip] = useState(() => LOADING_QUIPS[Math.floor(Math.random() * LOADING_QUIPS.length)]);

  useEffect(() => {
    const interval = setInterval(() => {
      setQuip(LOADING_QUIPS[Math.floor(Math.random() * LOADING_QUIPS.length)]);
    }, 2500);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="typing-indicator">
      <div className="typing-dots">
        <span></span><span></span><span></span>
      </div>
      <span className="typing-quip">{quip}</span>
    </div>
  );
}

function MessageBubble({ message, isUser, isStreaming, onRetry }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(message.content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className={`message-row ${isUser ? 'user-row' : 'bot-row'}`}>
      {!isUser && (
        <div className="avatar bot-avatar" title="The Judge">⚖️</div>
      )}
      <div className={`message-bubble ${isUser ? 'user-bubble' : 'bot-bubble'} ${isStreaming ? 'streaming' : ''}`}>
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <>
            <ReactMarkdown>{message.content + (isStreaming ? ' ▊' : '')}</ReactMarkdown>
            {!isStreaming && message.content && !message.isError && (
              <div className="bubble-actions">
                <button className="bubble-action-btn" onClick={handleCopy} title="Copy response">
                  {copied ? '✓' : '📋'}
                </button>
              </div>
            )}
          </>
        )}
        {message.judgment && (
          <div className="judgment-badge">{message.judgment}</div>
        )}
        {message.isError && onRetry && (
          <button className="retry-btn" onClick={onRetry}>🔄 Try Again</button>
        )}
      </div>
      {isUser && (
        <div className="avatar user-avatar" title="The Accused">🧑‍⚖️</div>
      )}
    </div>
  );
}

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [placeholder, setPlaceholder] = useState('');
  const [showWelcome, setShowWelcome] = useState(true);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [messageCount, setMessageCount] = useState(0);
  const [showDisclaimer, setShowDisclaimer] = useState(!localStorage.getItem('disclaimerAccepted'));
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Rotate placeholder text
  useEffect(() => {
    setPlaceholder(PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)]);
    const interval = setInterval(() => {
      setPlaceholder(PLACEHOLDERS[Math.floor(Math.random() * PLACEHOLDERS.length)]);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  // Health check with retry
  useEffect(() => {
    let retries = 0;
    const checkHealth = () => {
      fetch(`${API_BASE}/api/health`)
        .then(r => r.json())
        .then(() => setConnectionStatus('connected'))
        .catch(() => {
          if (retries < 3) {
            retries++;
            setConnectionStatus('checking');
            setTimeout(checkHealth, 2000);
          } else {
            setConnectionStatus('disconnected');
          }
        });
    };
    checkHealth();
  }, []);

  // Auto scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Auto-resize textarea
  const adjustTextarea = useCallback(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = Math.min(ta.scrollHeight, 150) + 'px';
    }
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handleGlobalKeys = (e) => {
      if (e.key === 'Escape' && (isLoading || isStreaming)) {
        handleStop();
      }
    };
    window.addEventListener('keydown', handleGlobalKeys);
    return () => window.removeEventListener('keydown', handleGlobalKeys);
  });

  const getRandomJudgment = () => JUDGMENTS[Math.floor(Math.random() * JUDGMENTS.length)];

  const handleStop = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
    setIsStreaming(false);
  };

  const sendMessage = async (retryMessage = null) => {
    const trimmed = retryMessage || input.trim();
    if (!trimmed || isLoading) return;

    setShowWelcome(false);

    // If retrying, remove the last error message
    let currentMessages = messages;
    if (retryMessage) {
      currentMessages = messages.filter(m => !m.isError);
      setMessages(currentMessages);
    }

    const userMsg = { role: 'user', content: trimmed };
    const newMessages = retryMessage
      ? [...currentMessages]
      : [...currentMessages, userMsg];

    if (!retryMessage) {
      setMessages(newMessages);
    }

    setInput('');
    setIsLoading(true);
    setMessageCount(prev => prev + 1);

    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    // Build history for API (last 20 messages for context window)
    const history = newMessages.slice(-20).map(m => ({
      role: m.role === 'assistant' ? 'assistant' : 'user',
      content: m.content,
    }));

    // Create abort controller for this request
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch(`${API_BASE}/api/chat/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: trimmed,
          history: history.slice(0, -1),
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({ error: `Server returned ${response.status}` }));
        throw new Error(errData.error || 'Server error');
      }

      setIsLoading(false);
      setIsStreaming(true);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let botContent = '';
      const botMsg = { role: 'assistant', content: '', judgment: null };

      setMessages(prev => [...prev, botMsg]);

      let buffer = '';
      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              if (data.token) {
                botContent += data.token;
                // eslint-disable-next-line no-loop-func
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    content: botContent,
                  };
                  return updated;
                });
              }
              if (data.done) {
                // eslint-disable-next-line no-loop-func
                setMessages(prev => {
                  const updated = [...prev];
                  updated[updated.length - 1] = {
                    ...updated[updated.length - 1],
                    content: botContent,
                    judgment: getRandomJudgment(),
                  };
                  return updated;
                });
              }
              if (data.error) {
                throw new Error(data.error);
              }
            } catch (parseErr) {
              if (parseErr.message !== 'Unexpected end of JSON input' &&
                  !parseErr.message.includes('JSON')) {
                throw parseErr;
              }
            }
          }
        }
      }

      setIsStreaming(false);
    } catch (err) {
      if (err.name === 'AbortError') {
        // User cancelled - keep what we have
        setMessages(prev => {
          const updated = [...prev];
          if (updated.length > 0 && updated[updated.length - 1].role === 'assistant') {
            updated[updated.length - 1] = {
              ...updated[updated.length - 1],
              content: updated[updated.length - 1].content + '\n\n*[Response stopped by user]*',
              judgment: '⏹️ STOPPED',
            };
          }
          return updated;
        });
      } else {
        setMessages(prev => [
          ...prev.filter(m => !(m.role === 'assistant' && !m.content)),
          {
            role: 'assistant',
            content: `⚠️ **Oops!** ${err.message}\n\nThe court is experiencing technical difficulties. Hit retry or rephrase your case.`,
            judgment: null,
            isError: true,
            failedMessage: trimmed,
          },
        ]);
      }
      setIsLoading(false);
      setIsStreaming(false);
    }

    abortControllerRef.current = null;
    inputRef.current?.focus();
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const clearChat = () => {
    setMessages([]);
    setShowWelcome(true);
    setMessageCount(0);
    inputRef.current?.focus();
  };

  const exportChat = () => {
    const transcript = messages.map(m => {
      const role = m.role === 'user' ? '🧑‍⚖️ You' : '⚖️ Judge';
      const judgment = m.judgment ? ` [${m.judgment}]` : '';
      return `${role}${judgment}:\n${m.content}\n`;
    }).join('\n---\n\n');

    const disclaimer = `⚠️ DISCLAIMER: This chatbot is for ENTERTAINMENT ONLY. All responses are generated by AI and should NOT be taken as professional advice, legal guidance, or factual truth. Do not make important life decisions based on this chatbot's responses. ⚠️\n\n`;
    const header = `🔥 Am I The A**hole? - Chat Transcript 🔥\nDate: ${new Date().toLocaleString()}\nMessages: ${messages.length}\n${'═'.repeat(50)}\n\n`;
    const blob = new Blob([header + disclaimer + transcript], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `aita-verdict-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const acceptDisclaimer = () => {
    localStorage.setItem('disclaimerAccepted', 'true');
    setShowDisclaimer(false);
  };

  return (
    <div className="app">
      {/* Disclaimer Modal */}
      {showDisclaimer && (
        <div className="disclaimer-overlay">
          <div className="disclaimer-modal">
            <div className="disclaimer-icon">⚠️</div>
            <h2>Important: Read Before Using</h2>
            <div className="disclaimer-content">
              <p><strong>This chatbot is for ENTERTAINMENT ONLY.</strong></p>
              <p>All responses are generated by an AI language model and should <strong>NOT</strong> be considered as:</p>
              <ul>
                <li>Professional advice (legal, medical, financial, or otherwise)</li>
                <li>Factual truth or accurate information</li>
                <li>Guidance for making important life decisions</li>
                <li>A substitute for professional consultation</li>
              </ul>
              <p>The AI has <strong>no real-world experience</strong>, <strong>no moral authority</strong>, and can produce incorrect, biased, or nonsensical responses.</p>
              <p className="disclaimer-fun">This is a fun, silly chatbot. Treat it as such! 🎭</p>
            </div>
            <button className="disclaimer-accept" onClick={acceptDisclaimer}>
              I Understand — Let's Have Fun!
            </button>
          </div>
        </div>
      )}

      {/* Animated background blobs */}
      <div className="bg-blob blob-1"></div>
      <div className="bg-blob blob-2"></div>
      <div className="bg-blob blob-3"></div>

      {/* Header */}
      <header className="header">
        <div className="header-content">
          <h1 className="title">
            <span className="title-emoji">🔥</span>
            Am I The A**hole?
            <span className="title-emoji">🔥</span>
          </h1>
          <p className="subtitle">AI-Powered Moral Judgment &bull; Powered by Gemma 3</p>
          <div className="header-badges">
            <div className={`status-badge ${connectionStatus}`}>
              <span className="status-dot"></span>
              {connectionStatus === 'connected' ? 'Judge is IN' : connectionStatus === 'checking' ? 'Waking up...' : 'Judge is OUT'}
            </div>
            {messageCount > 0 && (
              <div className="msg-count-badge">
                {messageCount} {messageCount === 1 ? 'case' : 'cases'} heard
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Chat Area */}
      <main className="chat-area">
        <div className="messages-container">
          {showWelcome && (
            <div className="welcome-card">
              <div className="welcome-gavel">⚖️</div>
              <h2>Order in the Court!</h2>
              {WELCOME_MESSAGES.map((msg, i) => (
                <p key={i}>{msg}</p>
              ))}
              <div className="starter-chips">
                {[
                  "Am I wrong for...",
                  "Judge me: I...",
                  "My friend says I'm the AH because...",
                  "Rate my pettiness:",
                ].map((chip, i) => (
                  <button
                    key={i}
                    className="starter-chip"
                    onClick={() => {
                      setInput(chip);
                      inputRef.current?.focus();
                    }}
                  >
                    {chip}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((msg, idx) => (
            <MessageBubble
              key={`${idx}-${msg.content.slice(0, 20)}`}
              message={msg}
              isUser={msg.role === 'user'}
              isStreaming={isStreaming && idx === messages.length - 1 && msg.role === 'assistant'}
              onRetry={msg.isError ? () => sendMessage(msg.failedMessage) : null}
            />
          ))}

          {isLoading && (
            <div className="message-row bot-row">
              <div className="avatar bot-avatar">⚖️</div>
              <div className="message-bubble bot-bubble">
                <TypingIndicator />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="input-area">
        <div className="input-container">
          <div className="input-left-actions">
            {messages.length > 0 && (
              <>
                <button className="action-btn clear-btn" onClick={clearChat} title="New Case">
                  🗑️
                </button>
                <button className="action-btn export-btn" onClick={exportChat} title="Export Transcript">
                  📥
                </button>
              </>
            )}
          </div>
          <textarea
            ref={(el) => { textareaRef.current = el; inputRef.current = el; }}
            className="chat-input"
            value={input}
            onChange={(e) => { setInput(e.target.value); adjustTextarea(); }}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            rows={1}
            disabled={isLoading || isStreaming}
            aria-label="Type your message"
          />
          {(isLoading || isStreaming) ? (
            <button
              className="send-btn stop-btn active"
              onClick={handleStop}
              title="Stop (Esc)"
            >
              <span className="send-icon">⏹</span>
            </button>
          ) : (
            <button
              className={`send-btn ${input.trim() ? 'active' : ''}`}
              onClick={() => sendMessage()}
              disabled={!input.trim() || isLoading || isStreaming}
              title="Submit for Judgment (Enter)"
            >
              <span className="send-icon">⚡</span>
            </button>
          )}
        </div>
        <div className="footer-disclaimer">
          <div className="disclaimer-text">
            <span className="disclaimer-warning">⚠️</span>
            <strong>ENTERTAINMENT ONLY:</strong> This AI has no real authority, experience, or factual accuracy. 
            Do not follow its advice for real-life decisions. Responses are for fun, not truth.
            <span className="disclaimer-warning">⚠️</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
