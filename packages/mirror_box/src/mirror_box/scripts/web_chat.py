"""Web UI for the Mirror Box — chat + live consciousness monitor.

Serves a single-page app with:
- Left panel: live metrics dashboard (R, surprise, C, state, timeline chart)
- Center panel: inner monologue stream (auto-generated self-reports)
- Right panel: chat with the system via IntrospectionDialogue

The conscious loop runs continuously in the background, pushing metrics
to all connected clients via WebSocket. Inner monologue is generated
periodically (every N frames) by the StateReporter.

Usage:
    python -m mirror_box.scripts.web_chat --gpu          # http://localhost:8420
    python -m mirror_box.scripts.web_chat --gpu --port 9000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from mirror_box.factory import create_mirror_box, gpu_config

logger = logging.getLogger(__name__)

METRICS_LOG_INTERVAL = 5  # log metrics every N frames (not every frame)


class SessionLogger:
    """Append-only JSONL logger for session data (metrics, monologue, chat)."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self._dir = Path(log_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self._path = self._dir / f"session_{ts}.jsonl"
        self._file = open(self._path, "a", encoding="utf-8")
        self._frame_counter = 0
        self.log("session_start", {})
        logger.info("Session log: %s", self._path)

    def log(self, event: str, data: dict) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **data,
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def log_metrics(self, data: dict) -> None:
        self._frame_counter += 1
        if self._frame_counter % METRICS_LOG_INTERVAL == 0:
            self.log("metrics", data)

    def log_monologue(self, subtype: str, text: str, frame: int,
                      R: float, state: str) -> None:
        self.log("monologue", {
            "subtype": subtype, "text": text,
            "frame": frame, "R": R, "state": state,
        })

    def log_chat(self, role: str, message: str) -> None:
        self.log("chat", {"role": role, "message": message})

    def close(self) -> None:
        self.log("session_end", {})
        self._file.close()


SENSORY_NOISE_LEN = 8       # tokens per frame
SENSORY_VOCAB_SIZE = 32000  # Mistral vocab range
MONOLOGUE_PROB = 1 / 45     # ~once every 45 frames on average (Poisson-like)


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mirror Box — Consciousness Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #0d1117; --card: #161b22; --border: #30363d;
    --text: #c9d1d9; --text-dim: #8b949e; --text-bright: #f0f6fc;
    --blue: #58a6ff; --green: #3fb950; --orange: #f0883e;
    --purple: #bc8cff; --red: #f85149; --cyan: #39d2c0;
    --wake: #3fb950; --sleep: #6e7681; --dream: #bc8cff; --altered: #f0883e;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Cascadia Code', 'Fira Code', monospace;
    background: var(--bg); color: var(--text);
    height: 100vh; overflow: hidden;
  }

  /* Header */
  .header {
    display: flex; align-items: center; gap: 16px;
    padding: 10px 20px; border-bottom: 1px solid var(--border);
    background: var(--card);
  }
  .header h1 { font-size: 16px; color: var(--text-bright); font-weight: 600; }
  .state-badge {
    padding: 3px 12px; border-radius: 12px; font-size: 12px;
    font-weight: 700; text-transform: uppercase; letter-spacing: 1px;
  }
  .state-wake { background: var(--wake); color: #000; }
  .state-sleep { background: var(--sleep); color: #000; }
  .state-dream { background: var(--dream); color: #000; }
  .state-altered { background: var(--altered); color: #000; }
  .header-info { font-size: 12px; color: var(--text-dim); }
  .header-controls { margin-left: auto; display: flex; gap: 6px; }
  .header-controls button {
    padding: 4px 12px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--card); color: var(--text); font-family: inherit;
    font-size: 11px; cursor: pointer; transition: all 0.15s;
  }
  .header-controls button:hover { background: var(--border); color: var(--text-bright); }
  .header-controls button.active { border-color: var(--blue); color: var(--blue); }

  /* Main layout */
  .main { display: grid; grid-template-columns: 300px 1fr 1fr; height: calc(100vh - 45px); }

  /* Left panel — metrics */
  .metrics-panel {
    border-right: 1px solid var(--border); padding: 12px;
    display: flex; flex-direction: column; gap: 10px; overflow-y: auto;
    min-height: 0;
  }
  .metric-card {
    background: var(--card); border-radius: 8px; padding: 10px 12px;
    border: 1px solid var(--border);
  }
  .metric-label { font-size: 10px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .metric-value { font-size: 28px; font-weight: 700; line-height: 1; }
  .metric-bar { height: 4px; border-radius: 2px; background: var(--border); margin-top: 6px; overflow: hidden; }
  .metric-bar-fill { height: 100%; border-radius: 2px; transition: width 0.3s ease; }
  .metric-sub { font-size: 11px; color: var(--text-dim); margin-top: 4px; }
  .chart-container { flex: 1; min-height: 200px; }
  .chart-container canvas { width: 100% !important; }
  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
  .metric-mini {
    background: var(--card); border-radius: 6px; padding: 8px;
    border: 1px solid var(--border);
  }
  .metric-mini .metric-label { font-size: 9px; margin-bottom: 3px; }
  .metric-mini .metric-value { font-size: 16px; }

  /* Center panel — inner monologue */
  .monologue-panel {
    border-right: 1px solid var(--border); display: flex; flex-direction: column;
    min-height: 0;
  }
  .panel-header {
    padding: 10px 14px; border-bottom: 1px solid var(--border);
    font-size: 11px; color: var(--text-dim); text-transform: uppercase;
    letter-spacing: 1px; background: var(--card);
  }
  .monologue-feed {
    flex: 1; overflow-y: auto; padding: 12px; display: flex;
    flex-direction: column; gap: 10px;
  }
  .monologue-entry {
    background: var(--card); border-radius: 8px; padding: 10px 12px;
    border-left: 3px solid var(--purple); font-size: 13px; line-height: 1.5;
    animation: fadeIn 0.5s ease;
  }
  .monologue-entry .meta {
    font-size: 10px; color: var(--text-dim); margin-bottom: 4px;
  }
  .monologue-entry .meta .state-tag {
    display: inline-block; padding: 1px 6px; border-radius: 4px;
    font-size: 9px; font-weight: 700; margin-left: 6px;
  }
  .monologue-entry.type-report { border-left-color: var(--cyan); }
  .monologue-entry.type-esm { border-left-color: var(--blue); }
  .monologue-entry.type-ewm { border-left-color: var(--green); }
  .monologue-entry.type-event { border-left-color: var(--orange); font-style: italic; }
  .thinking-indicator {
    display: none; padding: 8px 12px; font-size: 12px; color: var(--text-dim);
    animation: pulse 1.5s infinite;
  }
  .thinking-indicator.visible { display: block; }

  /* Right panel — chat */
  .chat-panel { display: flex; flex-direction: column; min-height: 0; }
  .chat-messages {
    flex: 1; overflow-y: auto; padding: 12px;
    display: flex; flex-direction: column; gap: 8px;
  }
  .msg {
    max-width: 90%; padding: 10px 14px; border-radius: 12px;
    font-size: 13px; line-height: 1.5; animation: fadeIn 0.3s ease;
  }
  .msg-user {
    align-self: flex-end; background: #1f3a5f; color: var(--text-bright);
    border-bottom-right-radius: 4px;
  }
  .msg-system {
    align-self: flex-start; background: var(--card); border: 1px solid var(--border);
    border-bottom-left-radius: 4px;
  }
  .msg-system .msg-label {
    font-size: 10px; color: var(--purple); font-weight: 600; margin-bottom: 4px;
  }
  .chat-input-area {
    padding: 12px; border-top: 1px solid var(--border); display: flex; gap: 8px;
  }
  .chat-input-area input {
    flex: 1; padding: 10px 14px; border-radius: 8px; border: 1px solid var(--border);
    background: var(--card); color: var(--text-bright); font-family: inherit;
    font-size: 13px; outline: none;
  }
  .chat-input-area input:focus { border-color: var(--blue); }
  .chat-input-area input::placeholder { color: var(--text-dim); }
  .chat-input-area button {
    padding: 10px 20px; border-radius: 8px; border: none;
    background: var(--blue); color: #000; font-family: inherit;
    font-weight: 600; font-size: 13px; cursor: pointer; transition: opacity 0.15s;
  }
  .chat-input-area button:hover { opacity: 0.85; }
  .chat-input-area button:disabled { opacity: 0.4; cursor: not-allowed; }

  @keyframes fadeIn { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: translateY(0); } }
  @keyframes pulse { 0%,100% { opacity: 0.4; } 50% { opacity: 1; } }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

  /* Settings modal */
  .modal-overlay {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.6);
    z-index: 100; align-items: center; justify-content: center;
  }
  .modal-overlay.visible { display: flex; }
  .modal {
    background: var(--card); border: 1px solid var(--border); border-radius: 12px;
    padding: 20px; width: 500px; max-width: 90vw; max-height: 80vh; overflow-y: auto;
  }
  .modal h2 { font-size: 14px; color: var(--text-bright); margin-bottom: 14px; }
  .modal label { font-size: 11px; color: var(--text-dim); text-transform: uppercase; letter-spacing: 1px; display: block; margin-bottom: 4px; }
  .modal textarea, .modal input[type="number"] {
    width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px;
    color: var(--text); font-family: inherit; font-size: 12px; padding: 8px; margin-bottom: 12px; resize: vertical;
  }
  .modal textarea { min-height: 80px; }
  .modal input[type="number"] { width: 80px; }
  .modal .row { display: flex; gap: 16px; align-items: center; margin-bottom: 12px; }
  .modal .row label { margin-bottom: 0; white-space: nowrap; }
  .modal .btn-row { display: flex; gap: 8px; justify-content: flex-end; margin-top: 8px; }
  .modal button {
    padding: 6px 16px; border-radius: 6px; border: 1px solid var(--border);
    background: var(--card); color: var(--text); font-family: inherit; font-size: 12px; cursor: pointer;
  }
  .modal button.primary { border-color: var(--blue); color: var(--blue); }
  .modal button:hover { background: var(--border); }
  .modal .hint { font-size: 10px; color: var(--text-dim); margin-top: -8px; margin-bottom: 12px; }
</style>
</head>
<body>

<div class="header">
  <h1>&#9679; Mirror Box</h1>
  <span id="stateBadge" class="state-badge state-wake">WAKE</span>
  <span id="headerInfo" class="header-info">Frame 0</span>
  <div class="header-controls">
    <button onclick="sendCmd('run', 20)">&#9654; Run 20</button>
    <button onclick="sendCmd('pause')" id="pauseBtn">&#9646;&#9646; Pause</button>
    <button onclick="sendCmd('stop')" id="stopBtn" style="border-color:var(--red);color:var(--red)">&#9632; Stop</button>
    <button onclick="sendCmd('restart')" style="border-color:var(--green);color:var(--green)">&#8635; Restart</button>
    <button onclick="sendCmd('state', 'wake')">Wake</button>
    <button onclick="sendCmd('state', 'sleep')">Sleep</button>
    <button onclick="sendCmd('state', 'dream')">Dream</button>
    <button onclick="sendCmd('state', 'altered')">Altered</button>
    <button onclick="openSettings()" title="Settings">&#9881;</button>
  </div>
</div>

<!-- Settings Modal -->
<div class="modal-overlay" id="settingsModal" onclick="if(event.target===this)closeSettings()">
  <div class="modal">
    <h2>&#9881; Settings</h2>
    <label>System Identity (prompt preamble)</label>
    <textarea id="identityInput" placeholder="Leave blank for unbiased self-knowledge. Example: You are Elsa, an artificial consciousness created by Matthias Gruber..."></textarea>
    <div class="hint">Injected into all generation prompts. Blank = neutral "You are an artificial consciousness system."</div>
    <label>Max Generation Tokens</label>
    <div class="row">
      <input type="number" id="maxTokensInput" value="48" min="16" max="256" step="8">
      <span style="font-size:11px;color:var(--text-dim)">Lower = faster, shorter responses</span>
    </div>
    <label>Monologue Interval (frames)</label>
    <div class="row">
      <input type="number" id="monologueIntervalInput" value="45" min="5" max="200" step="5">
      <span style="font-size:11px;color:var(--text-dim)">Higher = less frequent inner monologue</span>
    </div>
    <div class="btn-row">
      <button onclick="closeSettings()">Cancel</button>
      <button class="primary" onclick="saveSettings()">Apply</button>
    </div>
  </div>
</div>

<div class="main">
  <!-- Left: Metrics -->
  <div class="metrics-panel">
    <div class="metric-card">
      <div class="metric-label">Self-Knowledge (R)</div>
      <div class="metric-value" id="rValue" style="color:var(--blue)">0.000</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="rBar" style="width:0%;background:var(--blue)"></div></div>
      <div class="metric-sub" id="rDesc">Minimal — self-prediction is random</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Self-Surprise</div>
      <div class="metric-value" id="surpriseValue" style="color:var(--orange)">0.0</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="surpriseBar" style="width:0%;background:var(--orange)"></div></div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Conscious Content (C)</div>
      <div class="metric-value" id="cValue" style="color:var(--green)">0.000</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="cBar" style="width:0%;background:var(--green)"></div></div>
    </div>
    <div class="metric-card chart-container">
      <div class="metric-label">Timeline</div>
      <canvas id="timelineChart"></canvas>
    </div>
    <div class="metric-card">
      <div class="metric-label">Fatigue (KV Fill)</div>
      <div class="metric-value" id="fatigueValue" style="color:var(--red)">0.0%</div>
      <div class="metric-bar"><div class="metric-bar-fill" id="fatigueBar" style="width:0%;background:var(--red)"></div></div>
      <div class="metric-sub" id="fatigueDesc">Rested — working memory empty</div>
    </div>
    <div class="metric-card" id="sleepProgressCard" style="display:none">
      <div class="metric-label">Sleep Consolidation</div>
      <div class="metric-value" id="sleepProgressText" style="color:var(--sleep);font-size:16px">Epoch 0/0</div>
      <div class="metric-bar" style="height:6px"><div class="metric-bar-fill" id="sleepProgressBar" style="width:0%;background:var(--purple)"></div></div>
      <div class="metric-sub" id="sleepProgressLoss"></div>
    </div>
    <div class="metrics-grid">
      <div class="metric-mini">
        <div class="metric-label">Pred. Loss</div>
        <div class="metric-value" id="lossValue" style="color:var(--purple)">—</div>
      </div>
      <div class="metric-mini">
        <div class="metric-label">Complexity</div>
        <div class="metric-value" id="complexityValue" style="color:var(--cyan)">—</div>
      </div>
      <div class="metric-mini">
        <div class="metric-label">Gate World</div>
        <div class="metric-value" id="gateWorldValue">—</div>
      </div>
      <div class="metric-mini">
        <div class="metric-label">Gate Self</div>
        <div class="metric-value" id="gateSelfValue">—</div>
      </div>
    </div>
  </div>

  <!-- Center: Inner Monologue -->
  <div class="monologue-panel">
    <div class="panel-header">&#9672; Inner Monologue</div>
    <div class="monologue-feed" id="monologueFeed"></div>
    <div class="thinking-indicator" id="monologueThinking">generating inner monologue...</div>
  </div>

  <!-- Right: Chat -->
  <div class="chat-panel">
    <div class="panel-header">&#9656; Chat</div>
    <div class="chat-messages" id="chatMessages"></div>
    <div class="thinking-indicator" id="chatThinking">thinking...</div>
    <div class="chat-input-area">
      <input type="text" id="chatInput" placeholder="Ask me anything..."
             onkeydown="if(event.key==='Enter')sendChat()">
      <button id="sendBtn" onclick="sendChat()">Send</button>
    </div>
  </div>
</div>

<script>
const MAX_POINTS = 200;
const rData = [], surpriseData = [], lossData = [], labels = [];
let ws = null;
let paused = false;

// Chart setup
const ctx = document.getElementById('timelineChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: labels,
    datasets: [
      {
        label: 'R',
        data: rData,
        borderColor: '#58a6ff',
        backgroundColor: 'rgba(88,166,255,0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 2,
        yAxisID: 'y',
      },
      {
        label: 'Surprise',
        data: surpriseData,
        borderColor: '#f0883e',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 1.5,
        yAxisID: 'y2',
      },
      {
        label: 'Loss',
        data: lossData,
        borderColor: '#bc8cff',
        backgroundColor: 'transparent',
        tension: 0.3,
        pointRadius: 0,
        borderWidth: 1,
        borderDash: [4, 4],
        yAxisID: 'y2',
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: { legend: { display: true, labels: { color: '#8b949e', font: { size: 10 } } } },
    scales: {
      x: { display: false },
      y: {
        position: 'left',
        min: 0, max: 1,
        title: { display: true, text: 'R', color: '#58a6ff', font: { size: 10 } },
        ticks: { color: '#8b949e', font: { size: 9 } },
        grid: { color: 'rgba(48,54,61,0.5)' },
      },
      y2: {
        position: 'right',
        min: 0,
        title: { display: true, text: 'Surprise / Loss', color: '#f0883e', font: { size: 10 } },
        ticks: { color: '#8b949e', font: { size: 9 } },
        grid: { drawOnChartArea: false },
      },
    },
  },
});

function connect() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  ws = new WebSocket(`${proto}//${location.host}/ws`);
  ws.onopen = () => console.log('Connected');
  ws.onclose = () => { console.log('Disconnected'); setTimeout(connect, 2000); };
  ws.onmessage = (evt) => {
    const msg = JSON.parse(evt.data);
    if (msg.type === 'metrics') updateMetrics(msg);
    else if (msg.type === 'monologue') addMonologue(msg);
    else if (msg.type === 'chat_response') addChatResponse(msg.message);
    else if (msg.type === 'restart') clearPanels();
    else if (msg.type === 'state_change') updateStateBadge(msg.state);
    else if (msg.type === 'event') { addEvent(msg.message); if (msg.message.includes('consolidation complete') || msg.message.includes('consolidation error')) hideSleepProgress(); }
    else if (msg.type === 'sleep_progress') showSleepProgress(msg.epoch, msg.total_epochs, msg.loss);
    else if (msg.type === 'thinking') setThinking(msg.target, true);
    else if (msg.type === 'done_thinking') setThinking(msg.target, false);
    else if (msg.type === 'settings') {
      document.getElementById('identityInput').value = msg.identity || '';
      document.getElementById('maxTokensInput').value = msg.max_tokens;
      document.getElementById('monologueIntervalInput').value = msg.monologue_interval;
    }
  };
}

function updateMetrics(m) {
  const d = m.data;
  // Values
  document.getElementById('rValue').textContent = d.R.toFixed(3);
  document.getElementById('surpriseValue').textContent = d.surprise.toFixed(1);
  document.getElementById('cValue').textContent = d.C.toFixed(3);
  document.getElementById('lossValue').textContent = d.loss.toFixed(3);
  document.getElementById('complexityValue').textContent = d.complexity.toFixed(3);
  document.getElementById('gateWorldValue').textContent = d.gate_world.toFixed(2);
  document.getElementById('gateSelfValue').textContent = d.gate_self.toFixed(2);

  // Fatigue
  const fatPct = (d.fatigue * 100);
  document.getElementById('fatigueValue').textContent = fatPct.toFixed(1) + '%';
  document.getElementById('fatigueBar').style.width = fatPct + '%';
  let fatDesc = 'Rested';
  if (d.fatigue >= 0.9) fatDesc = 'Exhausted — sleep imminent';
  else if (d.fatigue >= 0.7) fatDesc = 'Tired — working memory filling';
  else if (d.fatigue >= 0.4) fatDesc = 'Moderate — accumulating experience';
  else if (d.fatigue >= 0.1) fatDesc = 'Light — some working memory used';
  document.getElementById('fatigueDesc').textContent = fatDesc;

  // Bars
  document.getElementById('rBar').style.width = (d.R * 100) + '%';
  document.getElementById('surpriseBar').style.width = Math.min(100, d.surprise / 60 * 100) + '%';
  document.getElementById('cBar').style.width = Math.min(100, d.C / 0.1 * 100) + '%';

  // R description
  let desc = 'Minimal — self-prediction is random';
  if (d.R >= 0.8) desc = 'High — strong self-knowledge';
  else if (d.R >= 0.5) desc = 'Moderate — partial self-prediction';
  else if (d.R >= 0.2) desc = 'Low — still learning about itself';
  document.getElementById('rDesc').textContent = desc;

  // State badge
  const badge = document.getElementById('stateBadge');
  badge.textContent = d.state.toUpperCase();
  badge.className = 'state-badge state-' + d.state;

  // Header info
  document.getElementById('headerInfo').textContent = `Frame ${d.frame}`;

  // Chart
  labels.push(d.frame);
  rData.push(d.R);
  surpriseData.push(d.surprise);
  lossData.push(d.loss);
  if (labels.length > MAX_POINTS) {
    labels.shift(); rData.shift(); surpriseData.shift(); lossData.shift();
  }
  chart.update();
}

function addMonologue(msg) {
  const feed = document.getElementById('monologueFeed');
  const el = document.createElement('div');
  const typeClass = 'type-' + (msg.subtype || 'report');
  el.className = 'monologue-entry ' + typeClass;
  const stateColors = { wake: 'var(--wake)', sleep: 'var(--sleep)', dream: 'var(--dream)', altered: 'var(--altered)' };
  const stateColor = stateColors[msg.state] || 'var(--text-dim)';
  el.innerHTML = `<div class="meta">Frame ${msg.frame} &middot; R=${msg.R}
    <span class="state-tag" style="background:${stateColor};color:#000">${(msg.state||'').toUpperCase()}</span></div>`
    + msg.text;
  feed.appendChild(el);
  feed.scrollTop = feed.scrollHeight;
}

function addEvent(text) {
  const feed = document.getElementById('monologueFeed');
  const el = document.createElement('div');
  el.className = 'monologue-entry type-event';
  el.textContent = text;
  feed.appendChild(el);
  feed.scrollTop = feed.scrollHeight;
}

function addChatResponse(text) {
  const messages = document.getElementById('chatMessages');
  const el = document.createElement('div');
  el.className = 'msg msg-system';
  el.innerHTML = '<div class="msg-label">Mirror Box</div>' + text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  setThinking('chat', false);
  document.getElementById('chatInput').disabled = false;
  document.getElementById('sendBtn').disabled = false;
}

function setThinking(target, on) {
  const el = document.getElementById(target + 'Thinking');
  if (el) el.classList.toggle('visible', on);
}

function clearPanels() {
  document.getElementById('monologueFeed').innerHTML = '';
  document.getElementById('chatMessages').innerHTML = '';
  hideSleepProgress();
  labels.length = 0; rData.length = 0; surpriseData.length = 0; lossData.length = 0;
  chart.update();
}

function updateStateBadge(state) {
  const badge = document.getElementById('stateBadge');
  badge.textContent = state.toUpperCase();
  badge.className = 'state-badge state-' + state;
}

function showSleepProgress(epoch, total, loss) {
  const card = document.getElementById('sleepProgressCard');
  card.style.display = '';
  document.getElementById('sleepProgressText').textContent = 'Epoch ' + epoch + '/' + total;
  document.getElementById('sleepProgressBar').style.width = (epoch / total * 100) + '%';
  document.getElementById('sleepProgressLoss').textContent = 'loss: ' + loss.toFixed(6);
}
function hideSleepProgress() {
  document.getElementById('sleepProgressCard').style.display = 'none';
}

function sendChat() {
  const input = document.getElementById('chatInput');
  const text = input.value.trim();
  if (!text || !ws) return;
  // Add user message
  const messages = document.getElementById('chatMessages');
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
  input.value = '';
  input.disabled = true;
  document.getElementById('sendBtn').disabled = true;
  ws.send(JSON.stringify({ type: 'chat', message: text }));
}

function sendCmd(cmd, arg) {
  if (!ws) return;
  if (cmd === 'pause') {
    paused = !paused;
    document.getElementById('pauseBtn').textContent = paused ? '\\u25b6 Resume' : '\\u23f8 Pause';
    document.getElementById('pauseBtn').classList.toggle('active', paused);
  }
  ws.send(JSON.stringify({ type: 'command', cmd: cmd, arg: arg }));
}

// Settings
function openSettings() {
  // Fetch current settings from server
  if (ws) ws.send(JSON.stringify({ type: 'get_settings' }));
  document.getElementById('settingsModal').classList.add('visible');
}
function closeSettings() {
  document.getElementById('settingsModal').classList.remove('visible');
}
function saveSettings() {
  if (!ws) return;
  ws.send(JSON.stringify({
    type: 'update_settings',
    identity: document.getElementById('identityInput').value,
    max_tokens: parseInt(document.getElementById('maxTokensInput').value) || 48,
    monologue_interval: parseInt(document.getElementById('monologueIntervalInput').value) || 45,
  }));
  closeSettings();
  addEvent('Settings updated');
}

connect();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="mirror-box-web", description="Mirror Box web UI")
    p.add_argument("--gpu", action="store_true", help="Use Mistral-7B FP16 on GPU")
    p.add_argument("--fast", action="store_true", help="Faster responses (shorter generation, less frequent monologue)")
    p.add_argument("--identity", type=str, default="", help="Identity preamble for prompts (blank = unbiased)")
    p.add_argument("--port", type=int, default=8420, help="Server port (default: 8420)")
    p.add_argument("--warmup", type=int, default=30, help="Warmup frames before serving")
    p.add_argument("--state-dim", type=int, default=None, help="Override state dimension")
    p.add_argument("--no-pause", action="store_true", help="Start running immediately (default: paused)")
    p.add_argument("--state", type=str, default="sleep", choices=["wake", "sleep", "dream", "altered"],
                   help="Starting conscious state (default: sleep)")
    return p.parse_args(argv)


class MirrorBoxServer:
    """FastAPI server wrapping the Mirror Box system."""

    def __init__(self, system: Any) -> None:
        self.system = system
        self.paused = False
        self.running = True
        self.clients: list[WebSocket] = []
        self.monologue_prob = MONOLOGUE_PROB
        self._bg_task: asyncio.Task | None = None
        self.session_log = SessionLogger()
        self._prev_state = "wake"  # track for sleep transition detection
        self._consolidating = False  # prevent re-entrant consolidation
        self.app = FastAPI(lifespan=self._lifespan)
        self._setup_routes()

    @asynccontextmanager
    async def _lifespan(self, app: FastAPI):
        self._bg_task = asyncio.create_task(self._frame_loop())
        yield
        self.running = False
        if self._bg_task:
            self._bg_task.cancel()
        self.session_log.close()

    def _setup_routes(self) -> None:
        app = self.app

        @app.get("/", response_class=HTMLResponse)
        async def index():
            return HTML_PAGE

        @app.websocket("/ws")
        async def websocket_endpoint(ws: WebSocket):
            await ws.accept()
            self.clients.append(ws)
            try:
                while True:
                    data = await ws.receive_json()
                    await self._handle_ws_message(ws, data)
            except WebSocketDisconnect:
                self.clients.remove(ws)
            except Exception:
                if ws in self.clients:
                    self.clients.remove(ws)

    async def _broadcast(self, msg: dict) -> None:
        """Send a message to all connected WebSocket clients."""
        data = json.dumps(msg)
        dead = []
        for ws in self.clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.remove(ws)

    async def _handle_ws_message(self, ws: WebSocket, data: dict) -> None:
        """Handle incoming WebSocket messages."""
        msg_type = data.get("type")

        if msg_type == "chat":
            question = data.get("message", "")
            self.session_log.log_chat("user", question)
            await self._broadcast({"type": "thinking", "target": "chat"})
            # Run a few frames to get fresh state, then generate
            for _ in range(3):
                await self._run_one_frame()
            response = await asyncio.to_thread(self.system.dialogue.ask, question)
            self.session_log.log_chat("system", response)
            await self._broadcast({
                "type": "done_thinking", "target": "chat",
            })
            await self._broadcast({
                "type": "chat_response", "message": response,
            })

        elif msg_type == "command":
            cmd = data.get("cmd")
            arg = data.get("arg")
            if cmd == "pause":
                self.paused = not self.paused
            elif cmd == "stop":
                self.paused = True
                await self._broadcast({
                    "type": "event", "message": "Stopped",
                })
            elif cmd == "restart":
                self.system.backend.clear_kv_cache()
                self.system.loop._frame_counter = 0
                self.paused = False
                self.session_log.log("restart", {})
                await self._broadcast({"type": "restart"})
                await self._broadcast({
                    "type": "event",
                    "message": "Restarted — working memory cleared",
                })
            elif cmd == "run":
                n = int(arg) if arg else 20
                for _ in range(n):
                    await self._run_one_frame()
            elif cmd == "state":
                from aiware.interfaces.data_types import ConsciousState
                state_map = {
                    "wake": ConsciousState.WAKE,
                    "sleep": ConsciousState.SLEEP,
                    "dream": ConsciousState.DREAM,
                    "altered": ConsciousState.ALTERED,
                }
                target = state_map.get(str(arg).lower())
                if target:
                    prev = self.system.state_ctrl.current_state().value
                    self.system.state_ctrl.transition_to(target)
                    self._prev_state = arg.lower()
                    self.session_log.log("state_change", {"to": arg.upper()})
                    await self._broadcast({
                        "type": "state_change",
                        "state": arg.lower(),
                    })
                    await self._broadcast({
                        "type": "event",
                        "message": f"State forced to {arg.upper()}",
                    })
                    # Trigger consolidation when manually entering sleep
                    if target == ConsciousState.SLEEP and prev != "sleep":
                        await self._run_sleep_consolidation()

        elif msg_type == "get_settings":
            avg_interval = int(1 / self.monologue_prob) if self.monologue_prob > 0 else 999
            await ws.send_json({
                "type": "settings",
                "identity": self.system.config.system_identity,
                "max_tokens": self.system.config.generation_max_tokens,
                "monologue_interval": avg_interval,
            })

        elif msg_type == "update_settings":
            identity = data.get("identity", "")
            max_tokens = int(data.get("max_tokens", 48))
            monologue_interval = int(data.get("monologue_interval", 45))
            # Update config
            self.system.config.generation_max_tokens = max_tokens
            self.system.config.system_identity = identity
            self.monologue_prob = 1 / max(monologue_interval, 1)
            # Propagate identity to reporter + dialogue
            self.system.reporter._identity = identity
            self.system.dialogue._identity = identity
            self.session_log.log("settings_change", {
                "identity": identity,
                "max_tokens": max_tokens,
                "monologue_avg_interval": monologue_interval,
            })

    async def _run_one_frame(self) -> dict | None:
        """Run a single conscious frame and broadcast metrics."""
        # Random noise as sensory input (every frame is different)
        noise = torch.randint(1, SENSORY_VOCAB_SIZE, (SENSORY_NOISE_LEN,))
        self.system.loop.set_sensory_input(noise)

        result = await self.system.loop.run_frame()

        # Build metrics payload
        state = self.system.state_ctrl.current_state()
        fatigue = self.system.backend.kv_fill_ratio()
        metrics = {
            "type": "metrics",
            "data": {
                "frame": result.frame_number,
                "R": round(result.R, 4),
                "surprise": round(result.self_surprise, 2),
                "C": round(result.C, 4),
                "complexity": round(result.complexity, 4),
                "loss": round(result.extras.get("prediction_loss", 0), 4),
                "gate_world": round(result.extras.get("gate_world_activation", 0), 3),
                "gate_self": round(result.extras.get("gate_self_activation", 0), 3),
                "fatigue": round(fatigue, 3),
                "state": state.value,
                "sleep_cycles": self.system.sleep_maintenance.total_cycles,
                "experience_buffer": self.system.sleep_maintenance.buffer_size,
            },
        }
        await self._broadcast(metrics)
        self.session_log.log_metrics(metrics["data"])

        # Record experience during wake (feeds sleep consolidation buffer)
        if state.value == "wake":
            self_state = self.system.loop.last_substrate_self_state
            if self_state is not None:
                telemetry = self.system.telemetry.collect()
                self.system.sleep_maintenance.record_experience(
                    telemetry, self_state,
                )

        # Detect transitions TO sleep (spontaneous or automatic)
        entering_sleep = False

        # Spontaneous sleep onset when working memory is exhausted
        if fatigue >= 0.95 and state.value == "wake":
            from aiware.interfaces.data_types import ConsciousState
            self.system.state_ctrl.transition_to(ConsciousState.SLEEP)
            self.session_log.log("spontaneous_sleep", {"fatigue": fatigue})
            await self._broadcast({
                "type": "state_change",
                "state": "sleep",
            })
            await self._broadcast({
                "type": "event",
                "message": "Spontaneous sleep onset — working memory exhausted",
            })
            entering_sleep = True

        # Automatic state-machine transition to sleep
        if not entering_sleep and self._prev_state == "wake" and state.value == "sleep":
            entering_sleep = True

        # Run sleep consolidation when entering sleep
        if entering_sleep and not self._consolidating:
            await self._run_sleep_consolidation()

        self._prev_state = state.value

        # Inner monologue — random trigger (Poisson-like)
        if random.random() < self.monologue_prob:
            asyncio.create_task(self._generate_monologue(result, state))

        return metrics

    async def _run_sleep_consolidation(self) -> None:
        """Run LoRA consolidation during sleep: replay, train, clear."""
        self._consolidating = True
        buf_size = self.system.sleep_maintenance.buffer_size
        if buf_size == 0:
            self._consolidating = False
            return

        await self._broadcast({
            "type": "event",
            "message": f"Sleep consolidation starting — replaying {buf_size} experiences...",
        })

        async def on_progress(epoch: int, total_epochs: int, loss: float) -> None:
            await self._broadcast({
                "type": "sleep_progress",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "loss": round(loss, 6),
            })

        try:
            result = await self.system.sleep_maintenance.run_maintenance(
                progress_callback=on_progress,
            )
            self.system.sleep_maintenance.clear_buffers()

            # Clear working memory (token buffer)
            self.system.backend.clear_kv_cache()

            self.session_log.log("sleep_consolidation", result)
            msg = (
                f"Sleep consolidation complete — "
                f"{int(result['total_steps'])} steps, "
                f"loss {result['initial_loss']:.4f} -> {result['final_loss']:.4f}"
            )
            await self._broadcast({"type": "event", "message": msg})
        except Exception as e:
            await self._broadcast({
                "type": "event",
                "message": f"Sleep consolidation error: {e}",
            })
        finally:
            self._consolidating = False

    async def _generate_monologue(self, result: Any, state: Any) -> None:
        """Generate and broadcast an inner monologue entry."""
        await self._broadcast({"type": "thinking", "target": "monologue"})

        # Random type selection
        subtype = random.choice(["report", "esm", "ewm"])
        generators = {
            "report": self.system.reporter.generate_state_report,
            "esm": self.system.reporter.generate_esm_narrative,
            "ewm": self.system.reporter.generate_ewm_narrative,
        }
        text = await asyncio.to_thread(generators[subtype])

        await self._broadcast({"type": "done_thinking", "target": "monologue"})
        self.session_log.log_monologue(
            subtype=subtype, text=text,
            frame=result.frame_number, R=result.R, state=state.value,
        )
        await self._broadcast({
            "type": "monologue",
            "subtype": subtype,
            "text": text,
            "frame": result.frame_number,
            "R": f"{result.R:.3f}",
            "state": state.value,
        })

    async def _frame_loop(self) -> None:
        """Background loop running conscious frames continuously."""
        while self.running:
            if self.paused or not self.clients:
                await asyncio.sleep(0.2)
                continue
            await self._run_one_frame()
            # Small yield to let other tasks run (WebSocket handling, etc.)
            await asyncio.sleep(0.01)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if not args.gpu:
        print("Error: --gpu is required (needs real LLM for text generation)")
        sys.exit(1)

    overrides = {}
    if args.state_dim is not None:
        overrides["state_dim"] = args.state_dim
    if args.identity:
        overrides["system_identity"] = args.identity
    if args.fast:
        overrides.setdefault("generation_max_tokens", 48)
    config = gpu_config(**overrides)

    mode = "FAST" if args.fast else "normal"
    print(f"Loading Mistral-7B FP16... [mode={mode}, max_tokens={config.generation_max_tokens}]")
    system = create_mirror_box(config)
    print(f"  State dim: {config.state_dim}, Device: {config.hardware.device}")

    # Warmup
    if args.warmup > 0:
        print(f"Warming up ({args.warmup} frames)...")
        import asyncio as _aio

        async def _warmup():
            for i in range(args.warmup):
                noise = torch.randint(1, SENSORY_VOCAB_SIZE, (SENSORY_NOISE_LEN,))
                system.loop.set_sensory_input(noise)
                await system.loop.run_frame()
                if i % 10 == 0:
                    latest = system.metrics_collector.get_latest()
                    if latest:
                        print(f"  Frame {latest.frame_number:3d} | R={latest.R:.3f}")
        _aio.run(_warmup())

    print(f"\nStarting web UI on http://localhost:{args.port}")
    print("Open in your browser to chat with Mirror Box\n")

    server = MirrorBoxServer(system)
    if args.fast:
        server.monologue_prob = 1 / 45
    if not args.no_pause:
        server.paused = True
        print("Starting PAUSED — unpause via UI or WebSocket")
    from aiware.interfaces.data_types import ConsciousState
    state_map = {"wake": ConsciousState.WAKE, "sleep": ConsciousState.SLEEP,
                 "dream": ConsciousState.DREAM, "altered": ConsciousState.ALTERED}
    system.state_ctrl.transition_to(state_map[args.state])
    print(f"Starting state: {args.state.upper()}")
    print(f"Session log: {server.session_log._path}")
    uvicorn.run(server.app, host="0.0.0.0", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
