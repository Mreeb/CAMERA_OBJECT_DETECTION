// Queue Monitoring Frontend Script

document.addEventListener("DOMContentLoaded", () => {
  loadQueueData();
  setInterval(loadQueueData, 1500);
});

async function loadQueueData() {
  try {
    const [qRes, jobsRes] = await Promise.all([
      fetch("/api/queue"),
      fetch("/api/jobs?limit=30"),
    ]);

    if (qRes.ok) {
      const q = await qRes.json();
      document.getElementById("q-waiting").textContent = q.waiting_clips || 0;
      document.getElementById("q-processing").textContent = q.processing_clips || 0;
      document.getElementById("q-completed").textContent = q.completed_clips || 0;
      document.getElementById("q-failed").textContent = `Failed: ${q.failed_clips || 0} clips`;
      document.getElementById("q-speed").textContent = `${q.avg_speed_ratio || 0.0}x`;
      document.getElementById("q-avg-time").textContent = `Avg time: ${q.avg_processing_time_sec || 0.0}s / clip`;
      document.getElementById("q-oldest-wait").textContent = q.oldest_waiting_created ? `Oldest: ${q.oldest_waiting_created.split(' ')[1]}` : "Oldest: None";
    }

    if (jobsRes.ok) {
      const jobs = await jobsRes.json();
      renderJobsTable(jobs);
    }
  } catch (err) {
    console.error("Failed to load queue data:", err);
  }
}

function renderJobsTable(jobs) {
  const tbody = document.getElementById("jobs-tbody");
  if (!jobs || jobs.length === 0) {
    tbody.innerHTML = `<tr><td colspan="9" style="text-align: center; color: var(--text-muted); padding: 2rem;">No jobs queued or processed yet.</td></tr>`;
    return;
  }

  tbody.innerHTML = jobs.map(j => {
    let badgeClass = "badge-stationary";
    if (j.status === "COMPLETED") badgeClass = "badge-moving";
    else if (j.status === "PROCESSING") badgeClass = "badge-low-motion";
    else if (j.status === "FAILED") badgeClass = "badge-ext-low";

    const procTimeStr = j.processing_time_sec ? `${j.processing_time_sec.toFixed(2)}s` : "-";
    const speedStr = j.speed_ratio ? `${j.speed_ratio.toFixed(1)}x real-time` : "-";
    const durStr = j.duration_sec ? `${j.duration_sec.toFixed(1)}s` : "-";

    return `
      <tr>
        <td style="font-weight: 700; color: var(--accent-cyan);">#${j.id}</td>
        <td><strong>${j.filename}</strong></td>
        <td><span style="color: var(--accent-amber);">${j.camera_id}</span></td>
        <td><span class="entity-badge ${badgeClass}">${j.status}</span></td>
        <td>${j.priority}</td>
        <td>${durStr}</td>
        <td>${procTimeStr}</td>
        <td style="color: var(--accent-green); font-weight: 600;">${speedStr}</td>
        <td>${j.attempts}</td>
      </tr>`;
  }).join("");
}
