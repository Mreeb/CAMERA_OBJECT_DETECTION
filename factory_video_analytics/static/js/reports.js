// Reports & Exports Frontend Script

document.addEventListener("DOMContentLoaded", () => {
  loadReport();
});

async function loadReport() {
  try {
    const res = await fetch("/api/shift-report");
    if (res.ok) {
      const data = await res.json();
      renderReport(data);
    }
  } catch (err) {
    console.error("Failed to load shift report:", err);
  }
}

function renderReport(data) {
  document.getElementById("report-narrative").textContent = data.plain_language_report || "No summary available.";

  const tbody = document.getElementById("metrics-tbody");
  const rows = [
    { label: "Shift Name", val: data.shift_name, desc: "Operational shift window" },
    { label: "Date", val: data.date_str, desc: "Reporting date" },
    { label: "Location", val: data.location_id, desc: "Physical facility zone" },
    { label: "Maximum Simultaneous People", val: data.max_people, desc: "Peak concurrent visible individuals" },
    { label: "Average Simultaneous People", val: data.avg_people, desc: "Mean visible count over observed time" },
    { label: "Total Observed Person-Hours", val: `${data.total_person_hours} hrs`, desc: "Cumulative visible worker time" },
    { label: "Likely Active Time", val: `${data.active_pct}% (${data.active_person_hours} hrs)`, desc: "Regular movement or physical handling" },
    { label: "Low Motion Time", val: `${data.low_motion_pct}% (${data.low_motion_person_hours} hrs)`, desc: "Stationary / standing (>30s)" },
    { label: "Unknown / Settling Time", val: `${data.unknown_pct}% (${data.unknown_person_hours} hrs)`, desc: "Occluded / initial track warm-up" },
    { label: "Confirmed Article Transfers", val: data.total_transfers, desc: "Boxes moved Storage -> Loading Area" },
    { label: "Line Crossings", val: data.total_line_crossings, desc: "Passage boundary crossings" },
    { label: "Extended Low Motion Events", val: data.ext_low_motion_count, desc: "Stationary events > 120s flagged for review" },
    { label: "Data Quality Audit", val: data.data_quality_warnings || "Optimal", desc: "Integrity and tracking continuity status" },
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td><strong>${r.label}</strong></td>
      <td style="color: var(--accent-cyan); font-weight: 600;">${r.val}</td>
      <td style="color: var(--text-muted); font-size: 0.80rem;">${r.desc}</td>
    </tr>
  `).join("");
}

async function generateAndExport() {
  try {
    const res = await fetch("/api/shift-report/generate", { method: "POST" });
    if (res.ok) {
      const result = await res.json();
      alert(`Reports exported successfully:\nTXT: ${result.exported_files.txt_path}\nJSON: ${result.exported_files.json_path}\nCSV: ${result.exported_files.csv_path}`);
      loadReport();
    }
  } catch (err) {
    alert(`Failed to export reports: ${err}`);
  }
}
