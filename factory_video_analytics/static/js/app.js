// Factory Vision AI - Pitch Black Minimalist Intelligence Script

let multiCameraChart = null;
let sparklineCharts = {};
let currentModalCameraId = null;
let currentStreamMode = "clean"; // "clean" or "ai"

// Cache for animated numbers
const numericState = {
  facility_score: 0,
  active_workers: 0,
  camera_01_score: 0,
  camera_02_score: 0,
  camera_03_score: 0,
  camera_04_score: 0,
  camera_01_moving: 0,
  camera_02_moving: 0,
  camera_03_moving: 0,
  camera_04_moving: 0,
  camera_01_low: 0,
  camera_02_low: 0,
  camera_03_low: 0,
  camera_04_low: 0,
};

document.addEventListener("DOMContentLoaded", () => {
  initMainChart();
  initSparklines();
  initLiveClock();
  fetchModelInfo();
  fetchAllData();
  setInterval(fetchAllData, 1000);
  setInterval(fetchModelInfo, 2500);
});

function initLiveClock() {
  const clockEl = document.getElementById("live-clock");
  setInterval(() => {
    const now = new Date();
    clockEl.textContent = now.toLocaleTimeString();
  }, 1000);
}

// Smooth Number Counter Animation with requestAnimationFrame
function animateNumber(elementId, targetValue, stateKey, suffix = "", duration = 350) {
  const el = document.getElementById(elementId);
  if (!el) return;

  const startValue = numericState[stateKey] || 0;
  if (startValue === targetValue) {
    el.textContent = `${Math.round(targetValue)}${suffix}`;
    return;
  }

  const startTime = performance.now();

  function updateCount(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1.0);
    // Ease out cubic
    const ease = 1 - Math.pow(1 - progress, 3);
    const current = startValue + (targetValue - startValue) * ease;

    el.textContent = `${Math.round(current)}${suffix}`;

    if (progress < 1.0) {
      requestAnimationFrame(updateCount);
    } else {
      numericState[stateKey] = targetValue;
      el.textContent = `${Math.round(targetValue)}${suffix}`;
    }
  }

  requestAnimationFrame(updateCount);
}

function initMainChart() {
  const ctx = document.getElementById("multiCameraActivityChart").getContext("2d");
  multiCameraChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [
        {
          label: "Camera 01 (Entrance)",
          data: [],
          borderColor: "#10b981",
          backgroundColor: "rgba(16, 185, 129, 0.05)",
          borderWidth: 2,
          tension: 0.45,
          fill: true,
          pointRadius: 0,
        },
        {
          label: "Camera 02 (Loading Yard)",
          data: [],
          borderColor: "#38bdf8",
          backgroundColor: "rgba(56, 189, 248, 0.05)",
          borderWidth: 2,
          tension: 0.45,
          fill: true,
          pointRadius: 0,
        },
        {
          label: "Camera 03 (Storage Bay)",
          data: [],
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.05)",
          borderWidth: 2,
          tension: 0.45,
          fill: true,
          pointRadius: 0,
        },
        {
          label: "Camera 04 (Dispatch Bay)",
          data: [],
          borderColor: "#a855f7",
          backgroundColor: "rgba(168, 85, 247, 0.05)",
          borderWidth: 2,
          tension: 0.45,
          fill: true,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 300, easing: "linear" },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
          backgroundColor: "#0d0d0d",
          borderColor: "#282828",
          borderWidth: 1,
          titleColor: "#ffffff",
          bodyColor: "#a1a1aa",
          titleFont: { family: "Plus Jakarta Sans", weight: "700" },
          bodyFont: { family: "Space Grotesk" },
        },
      },
      scales: {
        x: {
          grid: { color: "#141414" },
          ticks: { color: "#555555", font: { family: "Plus Jakarta Sans", size: 10 } },
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: "#141414" },
          ticks: {
            color: "#555555",
            font: { family: "Space Grotesk", size: 10 },
            callback: (v) => `${v}%`,
          },
        },
      },
    },
  });
}

function initSparklines() {
  const configs = [
    { id: "camera_01", color: "#10b981" },
    { id: "camera_02", color: "#38bdf8" },
    { id: "camera_03", color: "#f59e0b" },
    { id: "camera_04", color: "#a855f7" },
  ];

  configs.forEach(c => {
    const canvas = document.getElementById(`sparkline-${c.id}`);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    sparklineCharts[c.id] = new Chart(ctx, {
      type: "line",
      data: {
        labels: Array(15).fill(""),
        datasets: [
          {
            data: Array(15).fill(0),
            borderColor: c.color,
            backgroundColor: `${c.color}10`,
            borderWidth: 1.5,
            tension: 0.45,
            fill: true,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,
        plugins: { legend: { display: false }, tooltip: { enabled: false } },
        scales: {
          x: { display: false },
          y: { display: false, min: 0, max: 100 },
        },
      },
    });
  });
}

let currentChartFilter = "all";

function setChartFilter(filterId) {
  currentChartFilter = filterId;
  const pills = ["all", "camera_01", "camera_02", "camera_03", "camera_04"];
  pills.forEach(p => {
    const el = document.getElementById(`filter-${p}`);
    if (el) {
      el.classList.toggle("active", p === filterId);
    }
  });
  // Immediate fetch to switch view instantly
  fetchTimeline();
}

async function fetchTimeline() {
  try {
    const res = await fetch(`/api/activity/timeline?camera_id=${currentChartFilter}&max_points=35`);
    if (res.ok) {
      const timeline = await res.json();
      updateMainChart(timeline);
    }
  } catch (err) {
    console.warn("Timeline fetch error:", err);
  }
}

async function fetchAllData() {
  try {
    const [summaryRes, camerasRes, timelineRes, windowsRes] = await Promise.all([
      fetch("/api/activity/summary"),
      fetch("/api/activity/cameras"),
      fetch(`/api/activity/timeline?camera_id=${currentChartFilter}&max_points=35`),
      fetch("/api/activity/time-windows?limit=15"),
    ]);

    if (summaryRes.ok) {
      const summary = await summaryRes.json();
      renderExecutiveSummary(summary);
    }

    if (camerasRes.ok) {
      const cameras = await camerasRes.json();
      updateCameraCards(cameras);
    }

    if (timelineRes.ok) {
      const timeline = await timelineRes.json();
      updateMainChart(timeline);
    }

    if (windowsRes.ok) {
      const windows = await windowsRes.json();
      renderTimeWindows(windows);
    }
  } catch (err) {
    console.warn("Polling error:", err);
  }
}

function renderExecutiveSummary(s) {
  animateNumber("exec-facility-score", s.overall_activity_score, "facility_score", "%");
  animateNumber("exec-active-workers", s.total_active_workers_now, "active_workers");

  const levelEl = document.getElementById("exec-facility-level");
  levelEl.textContent = `Status: ${s.overall_activity_level}`;

  const scoreEl = document.getElementById("exec-facility-score");
  if (s.overall_activity_score >= 60) {
    scoreEl.style.color = "var(--accent-emerald)";
  } else if (s.overall_activity_score >= 25) {
    scoreEl.style.color = "var(--accent-cyan)";
  } else {
    scoreEl.style.color = "var(--text-dim)";
  }

  document.getElementById("exec-peak-window").textContent = s.peak_activity_window || "Calculating...";
  document.getElementById("exec-workers-sub").textContent = 
    `Moving: ${s.total_moving_workers_now || 0} | Stationary: ${s.total_low_motion_now || 0}`;
}

function updateCameraCards(cameras) {
  if (!cameras) return;

  cameras.forEach(cam => {
    const cid = cam.camera_id;
    const badgeEl = document.getElementById(`badge-${cid}`);
    const scoreEl = document.getElementById(`score-${cid}`);
    const meterEl = document.getElementById(`meter-${cid}`);

    // Update Activity Badge & Meter Color
    let badgeClass = "badge-idle";
    let accentColor = "var(--text-dim)";

    if (cam.activity_level === "HIGH ACTIVITY") {
      badgeClass = "badge-high";
      accentColor = "var(--accent-emerald)";
    } else if (cam.activity_level === "MODERATE") {
      badgeClass = "badge-mod";
      accentColor = "var(--accent-cyan)";
    } else if (cam.activity_level === "LOW MOTION") {
      badgeClass = "badge-low";
      accentColor = "var(--accent-amber)";
    }

    if (badgeEl) {
      badgeEl.className = `activity-badge ${badgeClass}`;
      badgeEl.textContent = cam.activity_level;
    }

    if (scoreEl) {
      scoreEl.style.color = accentColor;
      animateNumber(`score-${cid}`, cam.activity_score, `${cid}_score`, "%");
    }

    if (meterEl) {
      meterEl.style.width = `${cam.activity_score}%`;
      meterEl.style.background = accentColor;
    }

    // Smooth numbers for workers
    animateNumber(`moving-${cid}`, cam.moving_people, `${cid}_moving`);
    animateNumber(`low-${cid}`, cam.low_motion_people, `${cid}_low`);

    // Update Sparkline Chart in-place
    if (sparklineCharts[cid] && cam.sparkline && cam.sparkline.length > 0) {
      const spark = sparklineCharts[cid];
      spark.data.datasets[0].data = cam.sparkline;
      spark.update("none");
    }
  });
}

function updateMainChart(t) {
  if (!multiCameraChart || !t.labels || t.labels.length === 0) return;

  multiCameraChart.data.labels = t.labels;

  if (t.datasets) {
    multiCameraChart.data.datasets = t.datasets.map(ds => ({
      label: ds.label,
      data: ds.data,
      borderColor: ds.borderColor,
      backgroundColor: ds.backgroundColor,
      borderWidth: 2,
      tension: 0.45,
      fill: true,
      pointRadius: 0,
    }));
  }

  multiCameraChart.update("none");
}

function renderTimeWindows(windows) {
  const tbody = document.getElementById("time-windows-tbody");
  if (!windows || windows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align: center; color: var(--text-dim); padding: 2rem;">No peak activity windows detected yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = windows.map(w => {
    let badgeClass = "badge-idle";
    if (w.activity_level === "HIGH ACTIVITY") badgeClass = "badge-high";
    else if (w.activity_level === "MODERATE") badgeClass = "badge-mod";
    else if (w.activity_level === "LOW MOTION") badgeClass = "badge-low";

    return `
      <tr>
        <td><strong style="color: #fff; font-family: var(--font-mono);">${w.camera_id.toUpperCase()}</strong></td>
        <td><strong style="color: #ffffff;">${w.start_str} - ${w.end_str}</strong></td>
        <td style="color: var(--text-secondary); font-family: var(--font-mono);">${w.duration_str}</td>
        <td><span class="activity-badge ${badgeClass}">${w.activity_level}</span></td>
        <td style="font-weight: 700; color: #fff; font-family: var(--font-mono);">${w.avg_activity_score}%</td>
        <td><strong style="color: var(--accent-emerald); font-family: var(--font-mono);">${w.peak_workers}</strong></td>
        <td style="color: var(--text-secondary); font-size: 0.82rem;">${w.summary}</td>
      </tr>
    `;
  }).join("");
}

function openStreamModal(cameraId, cameraName) {
  currentModalCameraId = cameraId;
  document.getElementById("modal-camera-title").textContent = `Live Stream: ${cameraName}`;
  document.getElementById("modal-camera-sub").textContent = `Camera ID: ${cameraId.toUpperCase()} • 12 FPS • Landscape Orientation`;

  updateStreamSource();
  document.getElementById("live-stream-modal").style.display = "flex";
}

function closeStreamModal() {
  document.getElementById("live-stream-modal").style.display = "none";
  document.getElementById("modal-video-element").src = "";
  currentModalCameraId = null;
}

function setStreamMode(mode) {
  currentStreamMode = mode;
  document.getElementById("btn-mode-clean").classList.toggle("active", mode === "clean");
  document.getElementById("btn-mode-ai").classList.toggle("active", mode === "ai");
  updateStreamSource();
}

function updateStreamSource() {
  if (!currentModalCameraId) return;
  const img = document.getElementById("modal-video-element");
  img.src = `/stream/${currentModalCameraId}?overlay=${currentStreamMode}&t=${Date.now()}`;
}

// YOLO26 Scale & Confidence Switching System
let confDebounceTimer = null;

async function fetchModelInfo() {
  try {
    const res = await fetch("/api/models");
    if (res.ok) {
      const data = await res.json();
      updateModelUI(data.active_model, data.person_confidence, data.measured_latency_ms, data.measured_fps);
    }
  } catch (err) {
    console.warn("Error fetching model info:", err);
  }
}

function updateModelUI(activeModel, confidence, latencyMs, fps) {
  const select = document.getElementById("model-select");
  const badge = document.getElementById("model-status-badge");
  const perfBadge = document.getElementById("model-perf-badge");
  const confSlider = document.getElementById("conf-slider");
  const confBadge = document.getElementById("conf-val-badge");
  const modalPerf = document.getElementById("modal-perf-info");

  if (select && activeModel) {
    select.value = activeModel;
  }
  if (badge && activeModel) {
    if (activeModel.includes("factory_person") || activeModel.includes("best")) {
      badge.textContent = "★ SPECIALIZED";
      badge.classList.add("badge-specialized");
    } else {
      const shortScale = activeModel.replace(".pt", "").replace("yolo26", "YOLO26-").toUpperCase();
      badge.textContent = shortScale;
      badge.classList.remove("badge-specialized");
    }
  }

  // Update Live Measured FPS & Latency
  if (latencyMs !== undefined && latencyMs !== null && fps !== undefined && fps !== null) {
    const perfText = `⚡ ${latencyMs}ms • ${fps} FPS`;
    if (perfBadge) perfBadge.textContent = perfText;
    if (modalPerf) modalPerf.textContent = `⚡ Inference: ${latencyMs}ms • Model Capacity: ${fps} FPS`;
  }

  if (confidence !== undefined && confidence !== null) {
    const pct = Math.round(confidence * 100);
    if (confSlider) confSlider.value = pct;
    if (confBadge) confBadge.textContent = `${pct}%`;
  }
}

function handleConfidenceChange(val) {
  const confBadge = document.getElementById("conf-val-badge");
  if (confBadge) confBadge.textContent = `${val}%`;

  if (confDebounceTimer) clearTimeout(confDebounceTimer);
  confDebounceTimer = setTimeout(() => {
    updateConfidenceThreshold(parseFloat(val) / 100.0);
  }, 250);
}

async function updateConfidenceThreshold(confValue) {
  try {
    const res = await fetch("/api/models/confidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ confidence: confValue }),
    });
    const data = await res.json();
    if (res.ok && data.success) {
      showToast(`🎯 Detection Confidence: ${Math.round(data.person_confidence * 100)}%`, "success");
    } else {
      showToast(`❌ Failed to update confidence`, "error");
    }
  } catch (err) {
    console.error("Confidence update error:", err);
  }
}

async function switchModel(modelName) {
  showToast(`⏳ Loading ${modelName}...`, "info");
  try {
    const res = await fetch("/api/models/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: modelName }),
    });
    const data = await res.json();
    if (res.ok && data.success) {
      updateModelUI(data.active_model);
      showToast(`✅ Active Model: ${data.active_model.toUpperCase()}`, "success");
    } else {
      showToast(`❌ Failed to switch model: ${data.detail || "Error"}`, "error");
      fetchModelInfo();
    }
  } catch (err) {
    showToast(`❌ Network error while switching model`, "error");
    fetchModelInfo();
  }
}

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = "toast";
  toast.textContent = message;

  if (type === "error") {
    toast.style.borderColor = "#ef4444";
    toast.style.color = "#fca5a5";
  } else if (type === "success") {
    toast.style.borderColor = "var(--accent-emerald)";
    toast.style.color = "var(--accent-emerald)";
  }

  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
