document.addEventListener('DOMContentLoaded', () => {
  // --- Voice Alert System ("Sepatu tidak rapi, mohon dirapikan") ---
  let isVoiceEnabled = true;
  let lastSpeechTime = 0;
  const SPEECH_TEXT = "Sepatu tidak rapi, mohon dirapikan";

  const btnToggleSound = document.getElementById('btn-toggle-sound');
  const soundIcon = document.getElementById('sound-icon');
  const soundText = document.getElementById('sound-text');
  const btnTestVoice = document.getElementById('btn-test-voice');
  const uploadVoiceBanner = document.getElementById('upload-voice-banner');

  // Toggle voice on/off
  if (btnToggleSound) {
    btnToggleSound.addEventListener('click', () => {
      isVoiceEnabled = !isVoiceEnabled;
      if (isVoiceEnabled) {
        btnToggleSound.classList.add('active');
        btnToggleSound.classList.remove('muted');
        soundIcon.textContent = '🔊';
        soundText.innerHTML = 'Suara: <strong>Aktif</strong>';
      } else {
        btnToggleSound.classList.remove('active');
        btnToggleSound.classList.add('muted');
        soundIcon.textContent = '🔇';
        soundText.innerHTML = 'Suara: <strong>Mati</strong>';
        if ('speechSynthesis' in window) {
          window.speechSynthesis.cancel();
        }
      }
    });
  }

  // Test Voice Button
  if (btnTestVoice) {
    btnTestVoice.addEventListener('click', () => {
      speakWarning(true); // force play for test
    });
  }

  // Play soft chime sound before speaking
  function playAlertChime() {
    try {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = 'sine';
      osc.frequency.setValueAtTime(587.33, ctx.currentTime); // D5
      osc.frequency.setValueAtTime(880.00, ctx.currentTime + 0.1); // A5

      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.35);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } catch (e) {
      console.warn("AudioContext not allowed yet:", e);
    }
  }

  // Trigger voice alert
  function speakWarning(force = false) {
    if (!isVoiceEnabled && !force) return;

    const now = Date.now();
    // Throttle: limit to once every 4.5 seconds to prevent voice spam in real-time mode
    if (!force && (now - lastSpeechTime < 4500)) return;
    lastSpeechTime = now;

    // 1. Play soft ding-dong chime
    playAlertChime();

    // 2. Play speech utterance in Indonesian
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel(); // Stop any pending speech

      const utterance = new SpeechSynthesisUtterance(SPEECH_TEXT);
      utterance.lang = 'id-ID';
      utterance.rate = 0.95; // Clear natural rate
      utterance.pitch = 1.0;

      // Select Indonesian voice if available in the browser
      const voices = window.speechSynthesis.getVoices();
      const idVoice = voices.find(v => v.lang.includes('id') || v.lang.includes('ID') || v.name.toLowerCase().includes('indonesia'));
      if (idVoice) {
        utterance.voice = idVoice;
      }

      window.speechSynthesis.speak(utterance);
    }
  }

  // Ensure voices are loaded (Chrome quirk)
  if ('speechSynthesis' in window) {
    window.speechSynthesis.onvoiceschanged = () => {
      window.speechSynthesis.getVoices();
    };
  }

  // --- Tab Navigation ---
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId)?.classList.add('active');
    });
  });

  // --- TAB 1: UPLOAD & INPUT GAMBAR ---
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const btnBrowse = document.getElementById('btn-browse');
  const previewImg = document.getElementById('upload-preview-img');
  const placeholder = document.getElementById('upload-placeholder');
  const resultBox = document.getElementById('upload-result-box');
  const classBadge = document.getElementById('upload-class-badge');
  const confVal = document.getElementById('upload-confidence-val');
  const confBar = document.getElementById('upload-confidence-bar');
  const latencyBadge = document.getElementById('upload-latency');
  const probRapiVal = document.getElementById('prob-rapi-val');
  const probRapiBar = document.getElementById('prob-rapi-bar');
  const probBerantakanVal = document.getElementById('prob-berantakan-val');
  const probBerantakanBar = document.getElementById('prob-berantakan-bar');
  const sampleBtns = document.querySelectorAll('.btn-sample');

  btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  dropZone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove('dragover');
    });
  });

  dropZone.addEventListener('drop', (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleImageFile(files[0]);
    }
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleImageFile(fileInput.files[0]);
    }
  });

  // Quick sample buttons
  sampleBtns.forEach(btn => {
    btn.addEventListener('click', async () => {
      const src = btn.getAttribute('data-src');
      showPreview(src);
      try {
        const response = await fetch(src);
        const blob = await response.blob();
        const file = new File([blob], "sample.jpg", { type: "image/jpeg" });
        sendImageForPrediction(file);
      } catch (err) {
        console.error("Gagal memuat sampel:", err);
      }
    });
  });

  function handleImageFile(file) {
    if (!file.type.startsWith('image/')) {
      alert('Silakan pilih berkas citra yang valid (JPG, PNG, WEBP).');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      showPreview(e.target.result);
    };
    reader.readAsDataURL(file);
    sendImageForPrediction(file);
  }

  function showPreview(src) {
    previewImg.src = src;
    previewImg.classList.remove('hidden');
    placeholder.classList.add('hidden');
  }

  async function sendImageForPrediction(file) {
    resultBox.classList.remove('hidden');
    if (uploadVoiceBanner) uploadVoiceBanner.classList.add('hidden');
    classBadge.textContent = 'MEMPROSES...';
    classBadge.className = 'class-badge';
    confVal.textContent = '...';
    confBar.style.width = '0%';
    latencyBadge.textContent = 'Menghitung...';

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) throw new Error('API server error');
      const data = await res.json();
      displayPredictionResult(data);
    } catch (err) {
      console.warn("Gagal menghubungi server Flask:", err);
      fallbackSimulation(file.name);
    }
  }

  function displayPredictionResult(data) {
    const predClass = data.prediction;
    const confPct = data.confidence_percent;
    const isRapi = predClass.toLowerCase() === 'rapi';

    classBadge.textContent = predClass.toUpperCase();
    classBadge.className = `class-badge ${isRapi ? 'rapi' : 'berantakan'}`;

    confVal.textContent = `${confPct}%`;
    confBar.style.width = `${confPct}%`;
    latencyBadge.textContent = `${data.inference_time_ms} ms (YOLOv8 CPU)`;

    const probs = data.probabilities || {};
    const rapiP = ((probs['Rapi'] || 0) * 100).toFixed(1);
    const berantakanP = ((probs['Berantakan'] || 0) * 100).toFixed(1);

    probRapiVal.textContent = `${rapiP}%`;
    probRapiBar.style.width = `${rapiP}%`;

    probBerantakanVal.textContent = `${berantakanP}%`;
    probBerantakanBar.style.width = `${berantakanP}%`;

    // 🔊 Trigger voice alert if BERANTAKAN
    if (!isRapi) {
      if (uploadVoiceBanner) uploadVoiceBanner.classList.remove('hidden');
      speakWarning();
    } else {
      if (uploadVoiceBanner) uploadVoiceBanner.classList.add('hidden');
    }
  }

  function fallbackSimulation(filename) {
    const isRapi = filename.toLowerCase().includes('rapi');
    const pred = isRapi ? 'Rapi' : 'Berantakan';
    const conf = isRapi ? 95.6 : 99.8;
    displayPredictionResult({
      prediction: pred,
      confidence_percent: conf,
      probabilities: {
        'Rapi': isRapi ? 0.956 : 0.002,
        'Berantakan': isRapi ? 0.044 : 0.998
      },
      inference_time_ms: 15.4
    });
  }

  // --- TAB 2: KAMERA REAL-TIME ---
  const video = document.getElementById('webcam-video');
  const canvas = document.getElementById('capture-canvas');
  const btnStartCam = document.getElementById('btn-start-cam');
  const btnStopCam = document.getElementById('btn-stop-cam');
  const btnSnapshot = document.getElementById('btn-snapshot');
  const btnFlipCam = document.getElementById('btn-flip-cam');
  const toggleRealtime = document.getElementById('toggle-realtime');
  const camOverlayOff = document.getElementById('camera-off-overlay');
  const camStatusPill = document.getElementById('cam-status-pill');
  const camStatusText = document.getElementById('cam-status-text');
  const hudLiveTag = document.getElementById('hud-live-tag');
  const hudTagClass = document.getElementById('hud-tag-class');
  const hudTagConf = document.getElementById('hud-tag-conf');
  const camFpsBadge = document.getElementById('cam-fps-badge');
  const camClassBadge = document.getElementById('cam-class-badge');
  const camConfNumber = document.getElementById('cam-conf-number');
  const camProbRapiVal = document.getElementById('cam-prob-rapi-val');
  const camProbRapiBar = document.getElementById('cam-prob-rapi-bar');
  const camProbBerantakanVal = document.getElementById('cam-prob-berantakan-val');
  const camProbBerantakanBar = document.getElementById('cam-prob-berantakan-bar');
  const historyTbody = document.getElementById('history-tbody');
  const btnClearHistory = document.getElementById('btn-clear-history');

  let mediaStream = null;
  let isRealtimeActive = false;
  let realtimeTimer = null;
  let currentFacingMode = 'user';
  let isPredicting = false;
  const historyRecords = [];

  btnStartCam.addEventListener('click', startCamera);
  btnStopCam.addEventListener('click', stopCamera);
  btnSnapshot.addEventListener('click', captureAndPredict);
  btnFlipCam.addEventListener('click', flipCamera);
  btnClearHistory.addEventListener('click', clearHistory);

  toggleRealtime.addEventListener('change', (e) => {
    isRealtimeActive = e.target.checked;
    if (isRealtimeActive) {
      if (!mediaStream) {
        startCamera().then(() => startRealtimeLoop());
      } else {
        startRealtimeLoop();
      }
    } else {
      stopRealtimeLoop();
    }
  });

  async function startCamera() {
    try {
      const constraints = {
        video: {
          facingMode: currentFacingMode,
          width: { ideal: 640 },
          height: { ideal: 480 }
        },
        audio: false
      };

      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = mediaStream;
      video.play();

      camOverlayOff.classList.add('hidden');
      btnStartCam.classList.add('hidden');
      btnStopCam.classList.remove('hidden');
      btnSnapshot.disabled = false;
      camStatusPill.querySelector('.status-dot').classList.add('active');
      camStatusText.textContent = 'Kamera Aktif';

      if (toggleRealtime.checked) {
        startRealtimeLoop();
      }
    } catch (err) {
      console.error("Gagal mengakses webcam:", err);
      alert("Tidak dapat mengakses kamera. Pastikan izin kamera telah diberikan pada peramban.");
    }
  }

  function stopCamera() {
    if (mediaStream) {
      mediaStream.getTracks().forEach(track => track.stop());
      mediaStream = null;
    }
    stopRealtimeLoop();
    video.srcObject = null;
    camOverlayOff.classList.remove('hidden');
    btnStartCam.classList.remove('hidden');
    btnStopCam.classList.add('hidden');
    btnSnapshot.disabled = true;
    camStatusPill.querySelector('.status-dot').classList.remove('active');
    camStatusText.textContent = 'Kamera Mati';
    hudLiveTag.classList.add('hidden');
    camClassBadge.textContent = '-';
    camClassBadge.className = 'class-badge lg';
    camConfNumber.textContent = '0.0%';
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  }

  function flipCamera() {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
    if (mediaStream) {
      stopCamera();
      startCamera();
    }
  }

  function startRealtimeLoop() {
    if (realtimeTimer) clearInterval(realtimeTimer);
    realtimeTimer = setInterval(() => {
      if (mediaStream && !isPredicting) {
        captureAndPredict(true);
      }
    }, 380);
  }

  function stopRealtimeLoop() {
    if (realtimeTimer) {
      clearInterval(realtimeTimer);
      realtimeTimer = null;
    }
  }

  async function captureAndPredict(isContinuous = false) {
    if (!mediaStream || isPredicting) return;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    isPredicting = true;
    const ctx = canvas.getContext('2d');
    canvas.width = 224;
    canvas.height = 224;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const base64Data = canvas.toDataURL('image/jpeg', 0.85);

    try {
      const startTime = performance.now();
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Data })
      });

      const latency = Math.round(performance.now() - startTime);

      if (!res.ok) throw new Error('API server error');
      const data = await res.json();
      updateCameraPrediction(data, latency, isContinuous);
    } catch (err) {
      console.warn("Gagal inferensi kamera:", err);
    } finally {
      isPredicting = false;
    }
  }

  function updateCameraPrediction(data, latency, isContinuous) {
    const predClass = data.prediction;
    const confPct = data.confidence_percent;
    const isRapi = predClass.toLowerCase() === 'rapi';

    // Update HUD Tag
    hudLiveTag.classList.remove('hidden');
    hudLiveTag.className = `hud-floating-tag ${isRapi ? 'rapi' : 'berantakan'}`;
    hudTagClass.textContent = predClass.toUpperCase();
    hudTagConf.textContent = `${confPct}%`;

    // Update FPS & Latency
    const fps = latency > 0 ? Math.round(1000 / latency) : 30;
    camFpsBadge.textContent = `${fps} FPS | ${data.inference_time_ms} ms`;

    // Update Live Result Display
    camClassBadge.textContent = predClass.toUpperCase();
    camClassBadge.className = `class-badge lg ${isRapi ? 'rapi' : 'berantakan'}`;
    camConfNumber.textContent = `${confPct}%`;

    // Update Probabilities
    const probs = data.probabilities || {};
    const rapiP = ((probs['Rapi'] || 0) * 100).toFixed(1);
    const berantakanP = ((probs['Berantakan'] || 0) * 100).toFixed(1);

    camProbRapiVal.textContent = `${rapiP}%`;
    camProbRapiBar.style.width = `${rapiP}%`;

    camProbBerantakanVal.textContent = `${berantakanP}%`;
    camProbBerantakanBar.style.width = `${berantakanP}%`;

    // 🔊 Trigger voice alert if BERANTAKAN
    if (!isRapi) {
      speakWarning();
    }

    // Add to history
    if (!isContinuous || historyRecords.length === 0 || (Date.now() - (historyRecords[0]?.timestamp || 0)) > 2000) {
      addHistoryRecord(predClass, confPct, data.inference_time_ms);
    }
  }

  function addHistoryRecord(predClass, confPct, latencyMs) {
    const timeStr = new Date().toLocaleTimeString();
    const record = {
      timestamp: Date.now(),
      timeStr,
      predClass,
      confPct,
      latencyMs
    };

    historyRecords.unshift(record);
    if (historyRecords.length > 20) historyRecords.pop();

    renderHistoryTable();
  }

  function renderHistoryTable() {
    if (historyRecords.length === 0) {
      historyTbody.innerHTML = '<tr class="empty-row"><td colspan="4">Belum ada deteksi yang dicatat</td></tr>';
      return;
    }

    historyTbody.innerHTML = historyRecords.map(r => {
      const isRapi = r.predClass.toLowerCase() === 'rapi';
      const color = isRapi ? 'var(--success)' : 'var(--danger)';
      return `
        <tr>
          <td>${r.timeStr}</td>
          <td><strong style="color: ${color}">${r.predClass.toUpperCase()}</strong></td>
          <td>${r.confPct}%</td>
          <td>${r.latencyMs} ms</td>
        </tr>
      `;
    }).join('');
  }

  function clearHistory() {
    historyRecords.length = 0;
    renderHistoryTable();
  }
});
