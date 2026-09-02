const API_BASE = "http://127.0.0.1:8000";
const form = document.querySelector("#check-form");
const urlInput = document.querySelector("#url");
const htmlInput = document.querySelector("#html");
const message = document.querySelector("#form-message");
const featureList = document.querySelector("#feature-list");

const percent = value => `${(value * 100).toFixed(1)}%`;
const label = name => name.replaceAll("_", " ").replaceAll("registeration", "registration").toUpperCase();

function renderMetrics(data) {
  for (const [key, id] of [["accuracy", "metric-accuracy"], ["precision", "metric-precision"], ["recall", "metric-recall"], ["f1", "metric-f1"], ["roc_auc", "metric-roc"]]) document.querySelector(`#${id}`).textContent = percent(data[key]);
  const matrix = data.confusion_matrix.flat();
  ["matrix-tl", "matrix-tr", "matrix-bl", "matrix-br"].forEach((id, index) => { document.querySelector(`#${id}`).textContent = matrix[index]; });
}

function renderFeatures(features) {
  featureList.innerHTML = Object.entries(features).map(([name, value]) => {
    const positive = value === 1;
    return `<div class="feature-item ${positive ? "" : "is-risk"}"><label title="${name}">${label(name)}</label><div class="feature-bar"><span style="width:${positive ? 100 : 28}%"></span></div><b>${positive ? "PASS" : "FLAG"}</b></div>`;
  }).join("");
}

async function loadMetrics() {
  try { const response = await fetch(`${API_BASE}/metrics`); if (!response.ok) throw new Error(); renderMetrics(await response.json()); document.querySelector("#api-status").textContent = "API ONLINE"; document.querySelector(".status-dot").style.background = "#59b88d"; }
  catch { document.querySelector("#api-status").textContent = "API OFFLINE"; document.querySelector(".status-dot").style.background = "#d65c48"; }
}

form.addEventListener("submit", async event => {
  event.preventDefault(); message.textContent = "ANALYZING...";
  try {
    const response = await fetch(`${API_BASE}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url: urlInput.value.trim(), html: htmlInput.value || null }) });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Request failed");
    const risk = data.prediction === "phishing";
    document.querySelector("#result-empty").classList.add("hidden"); document.querySelector("#result-content").classList.remove("hidden");
    document.querySelector("#prediction").textContent = data.prediction.toUpperCase(); document.querySelector("#prediction").style.color = risk ? "var(--coral)" : "var(--teal)";
    document.querySelector("#probability").textContent = percent(data.phishing_probability); document.querySelector("#confidence").textContent = percent(data.confidence); document.querySelector("#confidence-fill").style.width = `${data.confidence * 100}%`; document.querySelector("#confidence-fill").style.background = risk ? "var(--coral)" : "var(--teal)";
    document.querySelector("#result-url").textContent = data.url; renderFeatures(data.features); message.textContent = "ASSESSMENT COMPLETE";
  } catch (error) { message.textContent = error.message.toUpperCase(); }
});

loadMetrics();