const connectButton = document.querySelector("#connectButton");
const talkButton = document.querySelector("#talkButton");
const stopButton = document.querySelector("#stopButton");
const connectionStatus = document.querySelector("#connectionStatus");
const transcript = document.querySelector("#transcript");
const sessionIdElement = document.querySelector("#sessionId");
const latencyElement = document.querySelector("#latency");
const hint = document.querySelector("#hint");
const composer = document.querySelector("#composer");
const messageInput = document.querySelector("#messageInput");
const sendButton = document.querySelector("#sendButton");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const canRecordAudio = Boolean(navigator.mediaDevices?.getUserMedia && window.MediaRecorder);

let socket;
let recognition;
let sessionId;
let recognitionStartedAt = 0;
let lastTurnStartedAt = 0;
let sessionTurnSeq = 0;
let speakingTurnId = null;
let mediaStream;
let mediaRecorder;
let audioChunks = [];

function createSessionId() {
  return `session_${crypto.randomUUID()}`;
}

function setConnectionState(state) {
  const connected = state === "Connected";
  const hasVoiceInput = Boolean(SpeechRecognition || canRecordAudio);
  connectionStatus.textContent = state;
  connectionStatus.dataset.state = state.toLowerCase();
  connectButton.disabled = connected;
  talkButton.disabled = !connected || !hasVoiceInput;
  stopButton.disabled = !connected;
  messageInput.disabled = !connected;
  sendButton.disabled = !connected;
}

function addMessage(role, text) {
  const message = document.createElement("article");
  message.className = `message ${role}`;

  const speaker = document.createElement("span");
  speaker.className = "speaker";
  speaker.textContent = role;

  const body = document.createElement("div");
  body.textContent = text;

  message.append(speaker, body);
  transcript.append(message);
  transcript.scrollTop = transcript.scrollHeight;
}

function sendEvent(event, payload = {}) {
  if (!socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }

  socket.send(JSON.stringify({ event, session_id: sessionId, payload }));
}

function stopSpeaking() {
  window.speechSynthesis.cancel();
  speakingTurnId = null;
}

function logLatency(turnId, label, ms, extra = {}) {
  const stamp = new Date().toISOString();
  console.log(`[latency ${stamp}] turn=${turnId} ${label}=${ms}ms`, extra);
}

function speak(text, turnId) {
  stopSpeaking();
  speakingTurnId = turnId;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;

  let ttsStartedAt = 0;
  utterance.onstart = () => {
    ttsStartedAt = performance.now();
  };
  utterance.onend = () => {
    if (speakingTurnId !== turnId) {
      return;
    }
    const ttsMs = Math.round(performance.now() - ttsStartedAt);
    logLatency(turnId, "tts_ms", ttsMs);
    speakingTurnId = null;
  };

  window.speechSynthesis.speak(utterance);
}

function handleInterruption(serverTurnSeq = null) {
  if (serverTurnSeq !== null) {
    sessionTurnSeq = Math.max(sessionTurnSeq, serverTurnSeq);
  } else {
    sessionTurnSeq += 1;
  }
  stopSpeaking();
}

function configureRecognition() {
  if (!SpeechRecognition) {
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  recognition.onresult = (event) => {
    let partialText = "";
    let finalText = "";

    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      if (result.isFinal) {
        finalText += result[0].transcript;
      } else {
        partialText += result[0].transcript;
      }
    }

    if (partialText) {
      hint.textContent = partialText;
      sendEvent("voice.user.transcript.partial", { text: partialText });
    }

    if (finalText) {
      const cleanedText = finalText.trim();
      const sttMs = Math.round(performance.now() - recognitionStartedAt);
      addMessage("user", cleanedText);
      lastTurnStartedAt = performance.now();
      logLatency(sessionTurnSeq + 1, "stt_ms", sttMs);
      sendEvent("voice.user.transcript.final", { text: cleanedText, stt_ms: sttMs });
    }
  };

  recognition.onend = () => {
    talkButton.disabled = !socket || socket.readyState !== WebSocket.OPEN;
    talkButton.textContent = "Start Talking";
  };
}

function selectRecordingMimeType() {
  const candidates = ["audio/ogg;codecs=opus", "audio/webm;codecs=opus", "audio/webm"];
  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || "";
}

function arrayBufferToBase64(arrayBuffer) {
  const bytes = new Uint8Array(arrayBuffer);
  let binary = "";
  const chunkSize = 0x8000;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

async function stopMediaRecording() {
  if (!mediaRecorder || mediaRecorder.state === "inactive") {
    return;
  }

  const recorder = mediaRecorder;
  const stopped = new Promise((resolve) => recorder.addEventListener("stop", resolve, { once: true }));
  recorder.stop();
  await stopped;

  const audioBlob = new Blob(audioChunks, { type: recorder.mimeType });
  audioChunks = [];
  mediaRecorder = null;
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaStream = null;

  if (!audioBlob.size) {
    hint.textContent = "No audio detected. Try again.";
    return;
  }

  const audioBuffer = await audioBlob.arrayBuffer();
  const sttMs = Math.round(performance.now() - recognitionStartedAt);
  sendEvent("voice.audio.chunk", {
    audio_base64: arrayBufferToBase64(audioBuffer),
    mime_type: audioBlob.type,
    is_final: true,
    stt_ms: sttMs,
  });
  hint.textContent = "Audio sent. Waiting for TriageOS...";
}

async function startMediaRecording() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const mimeType = selectRecordingMimeType();
    mediaRecorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
    audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size) {
        audioChunks.push(event.data);
      }
    });
    mediaRecorder.start();
    recognitionStartedAt = performance.now();
    talkButton.textContent = "Stop Talking";
    hint.textContent = "Listening... Press Stop Talking when you finish.";
  } catch (error) {
    addMessage("system", "Microphone access was not available. Check Firefox permissions and try again.");
    hint.textContent = "Microphone permission is required for voice input.";
    console.error(error);
  }
}

connectButton.addEventListener("click", () => {
  sessionId = createSessionId();
  sessionIdElement.textContent = sessionId;
  sessionTurnSeq = 0;
  configureRecognition();
  setConnectionState("Connecting");
  hint.textContent = "Connecting to the TriageOS voice engine...";

  const protocol = window.location.protocol === "https:" ? "wss" : "ws";
  const host = window.location.host || "localhost:8000";
  socket = new WebSocket(`${protocol}://${host}/v1/voice/sessions/${sessionId}/stream`);

  socket.onopen = () => {
    setConnectionState("Connected");
    hint.textContent = "Ready. Press Start Talking and say hello.";
  };

  socket.onclose = () => {
    setConnectionState("Disconnected");
    hint.textContent = "Session closed.";
  };

  socket.onmessage = (message) => {
    const serverEvent = JSON.parse(message.data);

    if (serverEvent.event === "voice.session.started") {
      addMessage("system", serverEvent.payload.message);
    }

    if (serverEvent.event === "voice.interruption.detected") {
      handleInterruption(serverEvent.payload.turn_seq ?? null);
      hint.textContent = "Interrupted — listening when you speak.";
    }

    if (serverEvent.event === "voice.assistant.response.created") {
      const turnId = serverEvent.payload.turn_id ?? 0;
      if (turnId < sessionTurnSeq) {
        logLatency(turnId, "stale_response_dropped", 0, { sessionTurnSeq });
        return;
      }

      const latency = serverEvent.payload.latency ?? {};
      const roundTripMs = Math.round(performance.now() - lastTurnStartedAt);
      latencyElement.textContent = `${roundTripMs} ms`;
      logLatency(turnId, "round_trip_ms", roundTripMs, latency);

      addMessage("assistant", serverEvent.payload.text);
      speak(serverEvent.payload.text, turnId);
    }

    if (serverEvent.event === "voice.error") {
      addMessage("system", serverEvent.payload.message);
    }

    if (serverEvent.event === "voice.audio.chunk.ack" && serverEvent.payload.transcribed === false) {
      addMessage("system", "Audio captured. Configure a speech-to-text provider to receive a transcript.");
    }
  };
});

talkButton.addEventListener("click", () => {
  if (SpeechRecognition) {
    if (!recognition) {
      return;
    }

    handleInterruption();
    sendEvent("voice.interruption.detected", {});
    talkButton.disabled = true;
    recognitionStartedAt = performance.now();
    hint.textContent = "Listening...";
    recognition.start();
    return;
  }

  if (mediaRecorder?.state === "recording") {
    stopMediaRecording();
    talkButton.textContent = "Start Talking";
    return;
  }

  handleInterruption();
  sendEvent("voice.interruption.detected", {});
  startMediaRecording();
});

composer.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text || !socket || socket.readyState !== WebSocket.OPEN) {
    return;
  }

  handleInterruption();
  sendEvent("voice.interruption.detected", {});
  addMessage("user", text);
  lastTurnStartedAt = performance.now();
  sendEvent("voice.user.transcript.final", { text, stt_ms: 0, source: "text" });
  messageInput.value = "";
  hint.textContent = "Message sent. Waiting for TriageOS...";
});

stopButton.addEventListener("click", () => {
  if (recognition) {
    recognition.stop();
  }

  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
  }
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaRecorder = null;
  mediaStream = null;

  handleInterruption();
  sendEvent("voice.session.ended", {});
  socket?.close();
});

setConnectionState("Disconnected");
