/* Zero build step on purpose: no bundler, no node_modules, no toolchain to break.
   The cost is that this is hand-written rather than componentised, which is a fair
   trade for a project meant to be cloned and run. */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};

const state = { sources: [], active: null };

const EXAMPLES = [
  "How do I return a custom response?",
  "What is dependency injection?",
  "How are background tasks run?",
  "What does the middleware documentation cover?",
];

/* Health drives everything the user sees. A running service in demo mode and one with
   retrieval enabled answer very differently, so the mode is shown, not assumed. */
async function refreshHealth() {
  const dot = $("health-dot");
  const meta = $("health-meta");
  try {
    const res = await fetch("/api/health");
    if (!res.ok) throw new Error(res.statusText);
    const h = await res.json();
    dot.className = "dot ok";
    const bits = [
      `mode=${h.mode}`,
      `retriever=${h.retriever}`,
      h.chunks ? `${h.chunks} chunks` : "no corpus",
      `providers=${(h.providers || []).join(",") || "none"}`,
    ];
    meta.textContent = bits.join(" · ");
  } catch (err) {
    dot.className = "dot bad";
    meta.textContent = `service unreachable (${err.message})`;
  }
}

/* Render the answer and turn every [n] into a clickable chip. A citation that does not
   resolve is styled as dangling rather than hidden: a broken marker in a UI implies
   support that was never established. */
function renderAnswer(payload) {
  const box = $("answer");
  box.replaceChildren();

  // One flowing paragraph, not one element per sentence. Splitting the answer into
  // paragraphs pushed every citation chip onto a line of its own, which reads as a
  // layout bug rather than as an annotated claim.
  const para = el("p");
  const parts = payload.text.split(/(\[\d+\])/g);

  for (const part of parts) {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) {
      if (part.trim()) para.appendChild(document.createTextNode(part));
      continue;
    }
    const index = Number(match[1]);
    const chip = el("span", "cite", String(index));
    const resolved = payload.sources.some((s) => s.index === index);
    if (!resolved) chip.classList.add("dangling");
    chip.title = resolved ? `Jump to source ${index}` : `Citation ${index} did not resolve`;
    chip.addEventListener("click", () => focusSource(index));
    para.appendChild(chip);
  }
  box.appendChild(para);

  $("demo-badge").hidden = !payload.is_demo;
  $("dangling").hidden = (payload.dangling_citations || []).length === 0;
  if (payload.dangling_citations?.length) {
    $("dangling").textContent =
      `Unresolved citations: ${payload.dangling_citations.join(", ")} — the model referenced a source that was not retrieved.`;
  }
}

function renderSources(payload) {
  const list = $("sources");
  list.replaceChildren();
  state.sources = payload.sources;

  for (const s of payload.sources) {
    const li = el("li", s.cited ? "cited" : null);
    li.dataset.index = String(s.index);

    const head = el("div", "s-head");
    head.appendChild(el("span", "s-doc", s.doc_id));
    head.appendChild(el("span", "score", s.score.toFixed(3)));
    li.appendChild(head);
    li.appendChild(el("div", "s-text", s.excerpt));
    li.addEventListener("click", () => focusCitation(s.index));

    list.appendChild(li);
  }

  if (!payload.sources.length) {
    list.appendChild(el("li", null, "No passages matched. The gateway will not answer without sources."));
  }
  $("coverage").textContent = `cited ${(payload.coverage * 100).toFixed(0)}%`;
}

/* The audit is the answer's trustworthiness made inspectable: how many claims were
   checked, how many failed, and whether the answer was withheld entirely. */
function renderAudit(payload) {
  const box = $("audit");
  const audit = payload.audit;
  if (!audit || !audit.verified) {
    box.hidden = true;
    return;
  }
  box.hidden = false;

  const parts = [
    `${audit.supported}/${audit.claims.length} claims grounded`,
    `verifier ${audit.verifier}`,
  ];
  if (audit.abstained) parts.push(`withheld (${payload.abstain_reason})`);
  if (audit.revisions) parts.push(`${audit.revisions} revision`);

  $("audit-text").textContent = parts.join(" · ");
  $("audit").className = "audit" + (audit.abstained ? " bad" : audit.unsupported ? " warn" : "");
}

function renderUsage(payload) {
  const dl = $("usage");
  dl.replaceChildren();
  const u = payload.usage || {};
  const rows = [
    ["retrieval", `${payload.retrieval_ms} ms`],
    ["generation", `${payload.generation_ms} ms`],
    ["model", u.model || "-"],
    ["prompt tokens", u.prompt_tokens ?? 0],
    ["completion tokens", u.completion_tokens ?? 0],
    ["total tokens", u.total_tokens ?? 0],
  ];
  for (const [k, v] of rows) {
    dl.appendChild(el("dt", null, k));
    dl.appendChild(el("dd", null, String(v)));
  }
}

/* Claims make partial failure visible. An answer can be supported on one point and
   fabricated on another, and a single answer-level verdict hides that. */
function renderClaims(payload) {
  const ul = $("claims");
  ul.replaceChildren();
  const byText = new Map((payload.audit?.claims || []).map((c) => [c.text, c]));
  for (const claim of payload.claims || []) {
    const verdict = byText.get(claim.text);
    const refs = claim.citations?.length ? claim.citations.join(", ") : "no citation";
    // An unsupported claim is the finding this panel exists to surface, so it is
    // marked in the same class as an uncited one rather than hidden.
    const bad = claim.uncited || (verdict && !verdict.supported);
    const li = el("li", bad ? "uncited" : null);
    li.textContent = verdict
      ? `${verdict.supported ? "supported" : "UNSUPPORTED"} — ${claim.text} [${refs}]`
      : `${claim.text} [${refs}]`;
    ul.appendChild(li);
  }
}

function focusSource(index) {
  const li = document.querySelector(`.sources li[data-index="${index}"]`);
  if (li) li.scrollIntoView({ behavior: "smooth", block: "nearest" });
  document.querySelectorAll(".cite").forEach((c) => c.classList.remove("active"));
  const chip = Array.from(document.querySelectorAll(".cite"))
    .find((c) => c.textContent === String(index));
  if (chip) chip.classList.add("active");
}

function focusCitation(index) {
  document.querySelectorAll(".sources li").forEach((li) => li.classList.remove("active"));
  const li = document.querySelector(`.sources li[data-index="${index}"]`);
  if (li) {
    li.classList.add("active");
    li.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
}

function setStatus(message, isError) {
  const node = $("status");
  if (!message) {
    node.hidden = true;
    return;
  }
  node.hidden = false;
  node.className = isError ? "status err" : "status";
  node.textContent = message;
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) {
    setStatus("Enter a question first.", true);
    return;
  }

  const button = $("go");
  button.disabled = true;
  setStatus("Retrieving and generating…");
  $("answer-wrap").hidden = true;

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, top_k: Number($("topk").value) || 5 }),
    });
    const data = await res.json();

    if (!res.ok) {
      // Surface the provider-level reason. A bare "request failed" makes a
      // misconfiguration indistinguishable from a model failure.
      const detail = data.providers
        ? Object.entries(data.providers).map(([k, v]) => `${k}: ${v}`).join("; ")
        : data.detail || data.error;
      throw new Error(`${res.status} — ${detail}`);
    }

    $("answer-wrap").hidden = false;
    renderAnswer(data);
    renderSources(data);
    renderUsage(data);
    renderClaims(data);
    renderAudit(data);
    setStatus("");
  } catch (err) {
    setStatus(err.message, true);
  } finally {
    button.disabled = false;
  }
}

function init() {
  const examples = $("examples");
  for (const text of EXAMPLES) {
    const b = el("button", null, text.length > 34 ? text.slice(0, 32) + "…" : text);
    b.title = text;
    b.addEventListener("click", () => {
      $("question").value = text;
      $("question").focus();
    });
    examples.appendChild(b);
  }

  $("go").addEventListener("click", ask);
  $("question").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) ask();
  });
  refreshHealth();
  setInterval(refreshHealth, 15000);
}

document.addEventListener("DOMContentLoaded", init);
