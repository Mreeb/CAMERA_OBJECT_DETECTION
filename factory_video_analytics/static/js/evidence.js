// Evidence Review Frontend Script

document.addEventListener("DOMContentLoaded", () => {
  loadEvidence();
});

async function loadEvidence() {
  const container = document.getElementById("evidence-container");
  try {
    const res = await fetch("/api/evidence?limit=40");
    if (res.ok) {
      const records = await res.json();
      renderEvidence(records);
    }
  } catch (err) {
    container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--accent-red);">Failed to load evidence: ${err}</div>`;
  }
}

function renderEvidence(records) {
  const container = document.getElementById("evidence-container");
  if (!records || records.length === 0) {
    container.innerHTML = `<div style="grid-column: 1 / -1; text-align: center; color: var(--text-muted); padding: 3rem;">No evidence records found.</div>`;
    return;
  }

  container.innerHTML = records.map(r => {
    let mediaHtml = "";
    if (r.evidence_clip_path) {
      mediaHtml = `<video controls class="evidence-media" src="${r.evidence_clip_path}" poster="${r.snapshot_path || ''}"></video>`;
    } else if (r.snapshot_path) {
      mediaHtml = `<img class="evidence-media" src="${r.snapshot_path}" alt="Event Snapshot">`;
    } else {
      mediaHtml = `<div class="evidence-media" style="display:flex;align-items:center;justify-content:center;color:var(--text-muted);">No Media</div>`;
    }

    const timeStr = r.datetime_str || `${r.timestamp.toFixed(1)}s`;
    let statusCol = "var(--text-muted)";
    if (r.human_verification === "confirmed") statusCol = "var(--accent-green)";
    else if (r.human_verification === "rejected") statusCol = "var(--accent-red)";

    return `
      <div class="evidence-card" id="ev-card-${r.id}">
        ${mediaHtml}
        <div class="evidence-body">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <strong style="color:var(--accent-cyan); font-size:0.90rem;">${r.event_type}</strong>
            <span style="font-size:0.75rem; color:${statusCol}; text-transform:uppercase; font-weight:700;">
              ${r.human_verification || 'unreviewed'}
            </span>
          </div>
          <div style="font-size:0.78rem; color:var(--text-secondary);">
            Time: ${timeStr} • Camera: <strong style="color:var(--text-primary);">${r.camera_id}</strong>
          </div>
          <div style="font-size:0.78rem; color:var(--text-secondary);">
            Session: <strong style="color:var(--accent-amber);">${r.session_id}</strong> (${r.class_name}) • Zone: ${r.zone || 'None'}
          </div>
          <div style="font-size:0.75rem; color:var(--text-muted); line-height:1.4;">
            ${r.details || ''}
          </div>
          <div style="display:flex; gap:0.5rem; margin-top:0.5rem;">
            <button class="btn-verify btn-confirm" onclick="verifyEvent(${r.id}, 'confirmed')">✓ Confirm</button>
            <button class="btn-verify btn-reject" onclick="verifyEvent(${r.id}, 'rejected')">✗ Reject</button>
          </div>
        </div>
      </div>`;
  }).join("");
}

async function verifyEvent(eventId, status) {
  try {
    const res = await fetch(`/api/evidence/${eventId}/verify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: status, notes: `Verified as ${status} by operator` }),
    });
    if (res.ok) {
      loadEvidence();
    }
  } catch (err) {
    alert(`Failed to verify: ${err}`);
  }
}
