"use strict";

const form = document.querySelector("#form");
const input = document.querySelector("#claim");
const button = document.querySelector("#argue");
const status = document.querySelector("#status");
const runStatus = document.querySelector("#run-status");
const support = document.querySelector("#support");
const oppose = document.querySelector("#oppose");
const foot = document.querySelector("#result-foot");
let active = false;

function setStatus(message, state = "") {
  status.textContent = message;
  status.className = "status " + state;
}

async function health() {
  try {
    const response = await fetch("./api/health", {cache: "no-store"});
    if (!response.ok) throw new Error("unavailable");
    const result = await response.json();
    setStatus(result.ready ? "Model online" : "Model loading", result.ready ? "online" : "");
  } catch {
    setStatus("Model unavailable", "offline");
  }
}

function empty(container, message) {
  container.replaceChildren();
  const p = document.createElement("p");
  p.className = "empty";
  p.textContent = message;
  container.append(p);
}

function displayEvidence(container, rows, side) {
  container.replaceChildren();
  if (!rows?.length) {
    empty(container, "No " + side + " evidence met the model's threshold. That doesn't prove there is none.");
    return;
  }
  for (const row of rows) {
    const card = document.createElement("article");
    card.className = "card";
    const head = document.createElement("div");
    head.className = "card-head";
    const ref = document.createElement("span");
    ref.className = "reference";
    ref.textContent = row.reference;
    const score = document.createElement("span");
    score.className = "score";
    const key = side === "support" ? "entailment" : "contradiction";
    const value = row.contextual?.[key];
    score.textContent = Number.isFinite(value) ? String(Math.round(value * 100)) + "% model score" : "Unscored";
    head.append(ref, score);
    const quotation = document.createElement("blockquote");
    quotation.textContent = row.passage;
    card.append(head, quotation);
    const details = document.createElement("details");
    const toggle = document.createElement("summary");
    toggle.textContent = "Context & score";
    const paragraph = document.createElement("p");
    paragraph.textContent = row.context || "No context available.";
    const signals = document.createElement("p");
    const isolated = row.isolated?.[key];
    signals.textContent = "Isolated passage: " + String(Math.round((isolated ?? 0) * 100)) + "% · Expanded context: " + String(Math.round((value ?? 0) * 100)) + "%";
    details.append(toggle, paragraph, signals);
    card.append(details);
    container.append(card);
  }
}

async function analyze(event) {
  event?.preventDefault();
  if (active) return;
  const claim = input.value.trim();
  if (!claim) return;
  active = true;
  button.disabled = true;
  button.textContent = "Consulting scripture…";
  runStatus.textContent = "Searching and judging passages…";
  empty(support, "Searching for the prosecution…");
  empty(oppose, "And the defense…");
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 45000);
  try {
    const response = await fetch("./api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({claim}),
      cache: "no-store",
      signal: controller.signal
    });
    const result = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(result.error || "Request failed (" + String(response.status) + ")");
    displayEvidence(support, result.for, "support");
    displayEvidence(oppose, result.against, "contradiction");
    runStatus.textContent = String(result.inspected ?? 0) + " candidate passages evaluated";
    foot.textContent = "These are the model's interpretations of selected Bible passages—not evidence about historical events or the truth of your claim. The score is not calibrated. Especially when it's impressive.";
  } catch (error) {
    runStatus.textContent = "Unable to finish this argument.";
    empty(support, error.name === "AbortError" ? "The model took too long. Try again." : error.message);
    empty(oppose, "No result.");
  } finally {
    clearTimeout(timer);
    active = false;
    button.disabled = false;
    button.textContent = "Make the case";
  }
}

form.addEventListener("submit", analyze);
document.querySelectorAll("[data-claim]").forEach(el => {
  el.addEventListener("click", () => {
    input.value = el.dataset.claim;
    form.requestSubmit();
  });
});
const query = new URL(location.href).searchParams.get("claim");
if (query && query.length <= 180) input.value = query;
health();
