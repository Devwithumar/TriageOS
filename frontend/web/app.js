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
const voiceSelect = document.querySelector("#voiceSelect");
const voiceStage = document.querySelector("#voiceStage");
const activityLabel = document.querySelector("#activityLabel");
const activityDetail = document.querySelector("#activityDetail");
const liveTranscript = document.querySelector("#liveTranscript");
const waveformBars = [...document.querySelectorAll("#waveform span")];

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
let reconnectTimer;
let reconnectAttempts = 0;
let intentionalClose = false;
let lastSocketError = false;
let audioContext;
let analyser;
let visualizerFrame;
let visualizerStream;
let availableVoices = [];

const maxReconnectAttempts = 5;

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
  if (state === "Connecting") {
    setActivity("connecting", "Connecting", "Opening a secure voice session...");
  } else if (state === "Reconnecting") {
    setActivity("reconnecting", "Reconnecting", `Retry ${reconnectAttempts} of ${maxReconnectAttempts}...`);
  } else if (state === "Disconnected") {
    setActivity("idle", "Ready when you are", "Connect to start a private voice session.");
  } else if (state === "Connected") {
    setActivity("ready", "Ready", "Press Start Talking or type a message.");
  }
}

function setActivity(mode, label, detail) {
  voiceStage.dataset.mode = mode;
  activityLabel.textContent = label;
  activityDetail.textContent = detail;
}

function setLiveTranscript(text = "") {
  liveTranscript.textContent = text;
  liveTranscript.hidden = !text;
}

function stopAudioVisualizer() {
  if (visualizerFrame) {
    window.cancelAnimationFrame(visualizerFrame);
    visualizerFrame = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  analyser = null;
  visualizerStream?.getTracks().forEach((track) => track.stop());
  visualizerStream = null;
  waveformBars.forEach((bar) => {
    bar.style.transform = "scaleY(0.25)";
  });
}

function startAudioVisualizer(stream) {
  stopAudioVisualizer();
  visualizerStream = stream;
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextClass) {
    return;
  }
  audioContext = new AudioContextClass();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  analyser.smoothingTimeConstant = 0.82;
  const source = audioContext.createMediaStreamSource(stream);
  source.connect(analyser);
  const samples = new Uint8Array(analyser.fftSize);

  const render = () => {
    analyser.getByteTimeDomainData(samples);
    const amplitude = Math.sqrt(samples.reduce((sum, value) => {
      const normalized = (value - 128) / 128;
      return sum + normalized * normalized;
    }, 0) / samples.length);
    waveformBars.forEach((bar, index) => {
      const centerDistance = Math.abs(index - (waveformBars.length - 1) / 2);
      const shape = 1 - centerDistance / (waveformBars.length / 2);
      const height = 0.22 + Math.min(1, amplitude * 8) * (0.35 + shape * 0.65);
      bar.style.transform = `scaleY(${height})`;
    });
    visualizerFrame = window.requestAnimationFrame(render);
  };
  render();
}

async function startMicrophoneVisualizer() {
  if (!canRecordAudio) {
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  startAudioVisualizer(stream);
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

function preferredVoice(voices) {
  const preferredNames = [
    "Microsoft Aria",
    "Microsoft Jenny",
    "Google US English",
    "Samantha",
    "Ava",
    "Karen",
  ];
  return preferredNames
    .map((name) => voices.find((voice) => voice.name.toLowerCase().includes(name.toLowerCase())))
    .find(Boolean) || voices.find((voice) => voice.lang.toLowerCase().startsWith("en"));
}

function refreshVoices() {
  availableVoices = window.speechSynthesis.getVoices().filter((voice) => voice.lang.toLowerCase().startsWith("en"));
  const selectedValue = voiceSelect.value || "auto";
  voiceSelect.replaceChildren(new Option("Natural voice (automatic)", "auto"));
  availableVoices.forEach((voice) => {
    voiceSelect.append(new Option(`${voice.name} · ${voice.lang}`, voice.name));
  });
  voiceSelect.value = availableVoices.some((voice) => voice.name === selectedValue) ? selectedValue : "auto";
}

window.speechSynthesis.addEventListener("voiceschanged", refreshVoices);
refreshVoices();

function logLatency(turnId, label, ms, extra = {}) {
  const stamp = new Date().toISOString();
  console.log(`[latency ${stamp}] turn=${turnId} ${label}=${ms}ms`, extra);
}

function speak(text, turnId) {
  stopSpeaking();
  speakingTurnId = turnId;

  const utterance = new SpeechSynthesisUtterance(text);
  const selectedVoice = voiceSelect.value === "auto"
    ? preferredVoice(availableVoices)
    : availableVoices.find((voice) => voice.name === voiceSelect.value);
  utterance.voice = selectedVoice || null;
  utterance.rate = 0.96;
  utterance.pitch = 1.04;
  utterance.volume = 0.9;

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
    setActivity("ready", "Ready", "Press Start Talking or type a message.");
    hint.textContent = "Ready for your next message.";
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
      setLiveTranscript(partialText);
      hint.textContent = "Listening...";
      sendEvent("voice.user.transcript.partial", { text: partialText });
    }

    if (finalText) {
      const cleanedText = finalText.trim();
      const sttMs = Math.round(performance.now() - recognitionStartedAt);
      addMessage("user", cleanedText);
      setLiveTranscript();
      lastTurnStartedAt = performance.now();
      setActivity("processing", "Processing", "Turning your words into a response...");
      hint.textContent = "Processing your message...";
      logLatency(sessionTurnSeq + 1, "stt_ms", sttMs);
      sendEvent("voice.user.transcript.final", { text: cleanedText, stt_ms: sttMs });
    }
  };

  recognition.onend = () => {
    talkButton.disabled = !socket || socket.readyState !== WebSocket.OPEN;
    talkButton.textContent = "Start Talking";
    stopAudioVisualizer();
    setActivity("ready", "Ready", "Press Start Talking or type a message.");
  };
}

function clearReconnectTimer() {
  if (reconnectTimer) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
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
  setActivity("processing", "Processing", "Transcribing your voice with Deepgram...");
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
    startAudioVisualizer(mediaStream);
    recognitionStartedAt = performance.now();
    talkButton.textContent = "Stop Talking";
    setActivity("listening", "Listening", "Speak naturally. Press Stop Talking when you finish.");
    hint.textContent = "Listening... Press Stop Talking when you finish.";
  } catch (error) {
    addMessage("system", "Microphone access was not available. Check Firefox permissions and try again.");
    hint.textContent = "Microphone permission is required for voice input.";
    console.error(error);
  }
}

function connect() {
  if (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) {
    return;
  }

  clearReconnectTimer();
  intentionalClose = false;
  lastSocketError = false;
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
    reconnectAttempts = 0;
    setConnectionState("Connected");
    hint.textContent = "Ready. Press Start Talking and say hello.";
  };

  socket.onclose = () => {
    socket = null;
    setConnectionState("Disconnected");
    if (intentionalClose) {
      hint.textContent = "Session ended.";
      return;
    }

    if (reconnectAttempts < maxReconnectAttempts) {
      reconnectAttempts += 1;
      setConnectionState("Reconnecting");
      hint.textContent = "Connection lost. Reconnecting automatically...";
      reconnectTimer = window.setTimeout(connect, Math.min(1000 * 2 ** (reconnectAttempts - 1), 8000));
    } else {
      hint.textContent = lastSocketError ? "Unable to reach the voice service." : "Session closed.";
      connectButton.disabled = false;
    }
  };

  socket.onerror = () => {
    lastSocketError = true;
  };

  socket.onmessage = (message) => {
    const serverEvent = JSON.parse(message.data);

    if (serverEvent.event === "voice.session.started") {
      setActivity("ready", "Ready", "Press Start Talking or type a message.");
    }

    if (serverEvent.event === "voice.interruption.detected") {
      handleInterruption(serverEvent.payload.turn_seq ?? null);
      hint.textContent = "Interrupted — listening when you speak.";
    }

    if (serverEvent.event === "voice.user.transcript.final") {
      addMessage("user", serverEvent.payload.text);
      setLiveTranscript();
      lastTurnStartedAt = performance.now();
      setActivity("processing", "Processing", "TriageOS is preparing a response...");
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
      setActivity("speaking", "Speaking", "TriageOS is responding.");
      hint.textContent = "TriageOS is speaking...";
      speak(serverEvent.payload.text, turnId);
    }

    if (serverEvent.event === "voice.error") {
      addMessage("system", serverEvent.payload.message);
      setActivity("error", "Something went wrong", "Check the message below and try again.");
    }

    if (serverEvent.event === "voice.audio.chunk.ack" && serverEvent.payload.transcribed === false) {
      addMessage("system", "Audio captured. Configure a speech-to-text provider to receive a transcript.");
    }
  };
}

connectButton.addEventListener("click", connect);

talkButton.addEventListener("click", () => {
  if (SpeechRecognition) {
    if (!recognition) {
      return;
    }

    handleInterruption();
    sendEvent("voice.interruption.detected", {});
    talkButton.disabled = true;
    recognitionStartedAt = performance.now();
    setActivity("listening", "Listening", "Speak naturally. I’m listening...");
    hint.textContent = "Listening...";
    startMicrophoneVisualizer()
      .catch((error) => console.warn("Microphone visualizer unavailable", error))
      .finally(() => recognition.start());
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
  setActivity("processing", "Processing", "TriageOS is preparing a response...");
  lastTurnStartedAt = performance.now();
  sendEvent("voice.user.transcript.final", { text, stt_ms: 0, source: "text" });
  messageInput.value = "";
  hint.textContent = "Message sent. Waiting for TriageOS...";
});

stopButton.addEventListener("click", () => {
  intentionalClose = true;
  clearReconnectTimer();
  if (recognition) {
    recognition.stop();
  }

  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
  }
  mediaStream?.getTracks().forEach((track) => track.stop());
  mediaRecorder = null;
  mediaStream = null;
  stopAudioVisualizer();

  handleInterruption();
  sendEvent("voice.session.ended", {});
  socket?.close();
});

setConnectionState("Disconnected");
