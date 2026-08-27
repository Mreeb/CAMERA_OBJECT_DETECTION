// Interactive Zone Calibration Canvas Script

const canvas = document.getElementById("calibration-canvas");
const ctx = canvas.getContext("2d");
const coordsEl = document.getElementById("canvas-coords");
const cameraId = "camera_02";

let bgImage = new Image();
let zones = [];
let currentDrawingPoints = [];
let isDrawing = false;

const ZONE_COLORS = {
  work: "#10b981",
  storage: "#f59e0b",
  loading: "#00d7ff",
  pickup: "#8b5cf6",
  destination: "#ec4899",
  exclusion: "#64748b"
};

document.addEventListener("DOMContentLoaded", async () => {
  await loadSnapshot();
  await loadExistingZones();
  setupCanvasEvents();
});

async function loadSnapshot() {
  return new Promise((resolve) => {
    bgImage.src = `/api/snapshot/${cameraId}?t=${Date.now()}`;
    bgImage.onload = () => {
      canvas.width = bgImage.width || 1280;
      canvas.height = bgImage.height || 576;
      redraw();
      resolve();
    };
    bgImage.onerror = () => {
      console.warn("Could not load camera snapshot. Using blank canvas.");
      resolve();
    };
  });
}

async function loadExistingZones() {
  try {
    const res = await fetch(`/api/zones/${cameraId}`);
    if (res.ok) {
      const data = await res.json();
      zones = data.zones || [];
      renderZonesList();
      redraw();
    }
  } catch (err) {
    console.error("Failed to load existing zones:", err);
  }
}

function setupCanvasEvents() {
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const x = Math.round((e.clientX - rect.left) * scaleX);
    const y = Math.round((e.clientY - rect.top) * scaleY);
    coordsEl.textContent = `X: ${x}, Y: ${y} (${(x / canvas.width).toFixed(2)}, ${(y / canvas.height).toFixed(2)})`;
  });

  canvas.addEventListener("click", (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    const normX = parseFloat(((e.clientX - rect.left) * scaleX / canvas.width).toFixed(3));
    const normY = parseFloat(((e.clientY - rect.top) * scaleY / canvas.height).toFixed(3));

    currentDrawingPoints.push([normX, normY]);
    redraw();
  });
}

function startDrawingZone() {
  const name = document.getElementById("zone-name").value.trim();
  if (!name) {
    alert("Please enter a Zone Name before drawing.");
    return;
  }
  isDrawing = true;
  currentDrawingPoints = [];
  redraw();
}

function completeCurrentZone() {
  if (!isDrawing || currentDrawingPoints.length < 3) {
    alert("Please place at least 3 points on the canvas before completing the zone.");
    return;
  }
  const name = document.getElementById("zone-name").value.trim() || "New Zone";
  const type = document.getElementById("zone-type").value;
  const id = `zone_${Date.now()}`;

  zones.push({
    id: id,
    name: name,
    type: type,
    polygon: currentDrawingPoints,
    color: hexToRgb(ZONE_COLORS[type] || "#ffffff"),
  });

  isDrawing = false;
  currentDrawingPoints = [];
  document.getElementById("zone-name").value = "";
  renderZonesList();
  redraw();
}

function cancelCurrentZone() {
  isDrawing = false;
  currentDrawingPoints = [];
  redraw();
}

function deleteZone(index) {
  zones.splice(index, 1);
  renderZonesList();
  redraw();
}

function renderZonesList() {
  const container = document.getElementById("zones-list-container");
  if (!zones || zones.length === 0) {
    container.innerHTML = `<div style="color: var(--text-muted); font-size: 0.8rem; padding: 0.5rem 0;">No zones defined</div>`;
    return;
  }

  container.innerHTML = zones.map((z, idx) => {
    const col = ZONE_COLORS[z.zone_type || z.type] || "#00d7ff";
    return `
      <div class="zone-item">
        <div style="display: flex; align-items: center; gap: 0.5rem;">
          <span style="width: 10px; height: 10px; border-radius: 2px; background: ${col}; display: inline-block;"></span>
          <strong>${z.name}</strong> <span style="color: var(--text-muted);">(${z.zone_type || z.type})</span>
        </div>
        <button style="background: none; border: none; color: var(--accent-red); cursor: pointer; font-size: 0.9rem;" onclick="deleteZone(${idx})">×</button>
      </div>`;
  }).join("");
}

function redraw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // 1. Draw Background Image
  if (bgImage.complete && bgImage.naturalWidth > 0) {
    ctx.drawImage(bgImage, 0, 0, canvas.width, canvas.height);
  } else {
    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  // 2. Draw Configured Zones
  zones.forEach(z => {
    const pts = z.polygon;
    if (!pts || pts.length < 3) return;

    const colHex = ZONE_COLORS[z.zone_type || z.type] || "#00d7ff";
    ctx.fillStyle = hexToRgba(colHex, 0.25);
    ctx.strokeStyle = colHex;
    ctx.lineWidth = 2;

    ctx.beginPath();
    ctx.moveTo(pts[0][0] * canvas.width, pts[0][1] * canvas.height);
    for (let i = 1; i < pts.length; i++) {
      ctx.lineTo(pts[i][0] * canvas.width, pts[i][1] * canvas.height);
    }
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    // Draw centroid label
    const avgX = pts.reduce((sum, p) => sum + p[0], 0) / pts.length * canvas.width;
    const avgY = pts.reduce((sum, p) => sum + p[1], 0) / pts.length * canvas.height;
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 13px Inter, sans-serif";
    ctx.fillText(z.name.toUpperCase(), avgX - 30, avgY);
  });

  // 3. Draw Currently Drawing Zone
  if (isDrawing && currentDrawingPoints.length > 0) {
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 4]);

    ctx.beginPath();
    ctx.moveTo(currentDrawingPoints[0][0] * canvas.width, currentDrawingPoints[0][1] * canvas.height);
    for (let i = 1; i < currentDrawingPoints.length; i++) {
      ctx.lineTo(currentDrawingPoints[i][0] * canvas.width, currentDrawingPoints[i][1] * canvas.height);
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw vertex dots
    currentDrawingPoints.forEach(p => {
      ctx.fillStyle = "#00d7ff";
      ctx.beginPath();
      ctx.arc(p[0] * canvas.width, p[1] * canvas.height, 5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

async function saveZonesAndCalibration() {
  const personConf = parseFloat(document.getElementById("slider-person-conf").value);
  const articleConf = parseFloat(document.getElementById("slider-article-conf").value);
  const statusEl = document.getElementById("save-status");

  try {
    const [zoneRes, calibRes] = await Promise.all([
      fetch(`/api/zones/${cameraId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ zones: zones }),
      }),
      fetch(`/api/calibration/${cameraId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompts: ["person", "cardboard box"],
          person_confidence: personConf,
          article_confidence: articleConf,
        }),
      }),
    ]);

    if (zoneRes.ok && calibRes.ok) {
      statusEl.style.display = "block";
      setTimeout(() => { statusEl.style.display = "none"; }, 3000);
    }
  } catch (err) {
    alert(`Failed to save calibration: ${err}`);
  }
}

function hexToRgb(hex) {
  const bigint = parseInt(hex.replace("#", ""), 16);
  return [(bigint >> 16) & 255, (bigint >> 8) & 255, bigint & 255];
}

function hexToRgba(hex, alpha) {
  const [r, g, b] = hexToRgb(hex);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}
