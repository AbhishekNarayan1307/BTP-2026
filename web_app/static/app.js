const form = document.querySelector("#ask-form");
const questionInput = document.querySelector("#question");
const domainSelect = document.querySelector("#domain");
const submitButton = document.querySelector("#submit");
const clearButton = document.querySelector("#clear");
const progress = document.querySelector("#progress");
const health = document.querySelector("#health");
const healthLabel = document.querySelector("#health-label");
const history = document.querySelector("#history");
const emptyState = document.querySelector("#empty-state");
const historyKey = "btp2026-evaluation-history";

function textElement(tag, className, value) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  element.textContent = value ?? "";
  return element;
}

function setHealth(payload) {
  health.classList.toggle("ready", payload.runtime_status === "ready");
  health.classList.toggle("failed", payload.runtime_status === "failed");
  const errors = Object.entries(payload.component_errors || {});
  if (payload.runtime_status === "ready") {
    const counts = Object.entries(payload.collections || {})
      .map(([domain, value]) => `${domain}: ${value.document_count}`)
      .join(" · ");
    healthLabel.textContent = `Ready · ${counts}`;
    health.title = `Initialization counts: ${JSON.stringify(payload.initialization_counts || {})}`;
  } else if (errors.length) {
    healthLabel.textContent = `Initialization failed · ${errors.map(([key]) => key).join(", ")}`;
    health.title = errors.map(([key, value]) => `${key}: ${value}`).join("\n");
  } else {
    healthLabel.textContent = "Backend is starting";
  }
}

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    setHealth(await response.json());
  } catch (error) {
    health.classList.add("failed");
    healthLabel.textContent = "Backend unavailable";
    health.title = error.message;
  }
}

function renderSource(source) {
  const item = textElement("li", "source");
  const title = source.source || source.act || source.paper_title || "Source";
  const page = source.page ?? source.pages ?? "Unknown";
  const info = [
    `Page: ${page}`,
    source.section && `Section: ${source.section}`,
    source.chunk_index && `Chunk: ${source.chunk_index}`,
    source.distance !== null && source.distance !== undefined && `Distance: ${Number(source.distance).toFixed(4)}`,
  ].filter(Boolean).join(" · ");
  item.append(textElement("div", "source-title", title));
  item.append(textElement("div", "", info));
  const metadata = source.metadata;
  if (metadata && Object.keys(metadata).length) {
    const meta = document.createElement("details");
    meta.append(textElement("summary", "", "Metadata"));
    meta.append(textElement("pre", "source-text", JSON.stringify(metadata, null, 2)));
    item.append(meta);
  }
  if (source.text) item.append(textElement("p", "source-text", source.text));
  return item;
}

function renderResult(data, persist = true) {
  emptyState?.remove();
  const evaluation = textElement("article", "evaluation");
  const heading = textElement("div", "evaluation-header");
  const titleArea = document.createElement("div");
  titleArea.append(textElement("h2", "", `${data.domain} evaluation`));
  titleArea.append(textElement("p", "evaluation-meta", data.question));
  heading.append(titleArea);
  heading.append(textElement("span", "total-time", `Total: ${Number(data.total_execution_seconds).toFixed(2)} s`));
  evaluation.append(heading);

  const grid = textElement("div", "model-grid");
  for (const result of data.results || []) {
    const card = textElement("section", "model-card");
    card.append(textElement("h3", "", result.model_name));
    if (result.status === "Success") card.append(textElement("p", "model-answer", result.answer));
    else card.append(textElement("p", "model-error", `${result.status}: ${result.error || "No answer returned."}`));

    const meta = textElement("div", "model-meta");
    meta.append(textElement("span", "pill", `${Number(result.execution_seconds).toFixed(2)} s`));
    if (result.retrieval_count !== null && result.retrieval_count !== undefined) {
      meta.append(textElement("span", "pill", `${result.retrieval_count} retrieved sources`));
    }
    if (result.verification_score !== null && result.verification_score !== undefined) {
      meta.append(textElement("span", "pill", `Support score: ${Number(result.verification_score).toFixed(2)}`));
    }
    if (result.abstained !== null && result.abstained !== undefined) {
      meta.append(textElement("span", `pill${result.abstained ? " abstained" : ""}`, `Abstained: ${result.abstained ? "Yes" : "No"}`));
    }
    if (result.status !== "Success") meta.append(textElement("span", "pill failed", result.status));
    card.append(meta);

    if (result.sources?.length) {
      const details = document.createElement("details");
      details.append(textElement("summary", "", `Retrieved sources (${result.sources.length})`));
      const list = textElement("ul", "source-list");
      for (const source of result.sources) list.append(renderSource(source));
      details.append(list);
      card.append(details);
    }
    if (result.verification) {
      const details = textElement("details", "verification");
      details.append(textElement("summary", "", "Verification details"));
      details.append(textElement("pre", "", JSON.stringify(result.verification, null, 2)));
      card.append(details);
    }
    grid.append(card);
  }
  evaluation.append(grid);
  history.prepend(evaluation);
  if (persist) {
    try {
      const saved = JSON.parse(sessionStorage.getItem(historyKey) || "[]");
      saved.push(data);
      sessionStorage.setItem(historyKey, JSON.stringify(saved));
    } catch (error) {
      console.warn("Could not save this evaluation in session history.", error);
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) {
    questionInput.focus();
    return;
  }
  submitButton.disabled = true;
  clearButton.disabled = true;
  domainSelect.disabled = true;
  progress.textContent = "Running all five models… this may take a while.";
  try {
    const response = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ domain: domainSelect.value, question }),
    });
    const payload = await response.json();
    if (!response.ok) {
      const detail = typeof payload.detail === "string" ? payload.detail : JSON.stringify(payload.detail, null, 2);
      throw new Error(detail || `Request failed with status ${response.status}`);
    }
    renderResult(payload);
    progress.textContent = "Evaluation complete.";
    await refreshHealth();
  } catch (error) {
    progress.textContent = `Request failed: ${error.message}`;
  } finally {
    submitButton.disabled = false;
    clearButton.disabled = false;
    domainSelect.disabled = false;
  }
});

clearButton.addEventListener("click", () => {
  questionInput.value = "";
  questionInput.focus();
});

refreshHealth();
window.setInterval(refreshHealth, 10000);
try {
  const saved = JSON.parse(sessionStorage.getItem(historyKey) || "[]");
  for (const result of saved) renderResult(result, false);
} catch (error) {
  console.warn("Could not restore session evaluation history.", error);
}
