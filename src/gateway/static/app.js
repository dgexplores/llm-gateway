/* Gateway UI.

   Zero build step on purpose: no bundler, no node_modules, no toolchain to break. The
   cost is that this is hand-written rather than componentised, which is a fair trade
   for a project meant to be cloned and run.

   One rule governs the rendering: a number must look like a number, and text the
   system did not verify must not look like text it did. */

const $ = (id) => document.getElementById(id);

const el = (tag, className, text) => {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
};

const reducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const state = { sources: [] };

const EXAMPLES = [
  "How do I return a custom response?",
  "What is dependency injection?",
  "How are background tasks run?",
  "How do I set a response status code?",
];

/* ── health ─────────────────────────────────────────────────────────────── */

/* Reports the mode, not just liveness. A running service in demo mode answers
   completely differently from one with retrieval and a model behind it, and a user
   cannot tell those apart from a green tick. */
async function refreshHealth() {
  const pill = $("health");
  const text = $("health-text");

  try {
    const res = await fetch("/api/health", { headers: { accept: "application/json" } });
    if (!res.ok) throw new Error(String(res.status));
    const h = await res.json();

    pill.dataset.state = "ok";
    const bits = [h.retriever];
    if (h.chunks) bits.push(`${h.chunks.toLocaleString()} chunks`);
    if (h.verifying) bits.push("verifying");
    if (h.providers?.length) bits.push(h.providers.join(" + "));
    text.textContent = bits.join(" · ");
    pill.title =
      `mode ${h.mode} · retriever ${h.retriever} · ${h.documents} documents · ` +
      `verification ${h.verifying ? "on" : "off"}`;
  } catch (err) {
    pill.dataset.state = "down";
    text.textContent = "unreachable";
    pill.title = `The service did not respond: ${err.message}`;
  }
}

/* ── answer ─────────────────────────────────────────────────────────────── */

function renderAnswer(payload) {
  const box = $("answer");
  box.replaceChildren();

  // One flowing paragraph. Splitting the answer into a block per sentence pushed
  // every citation onto a line of its own, which reads as a layout fault rather than
  // as an annotated claim.
  const para = el("p");
  const parts = payload.text.split(/(\[\d+\])/g);

  for (const part of parts) {
    const match = /^\[(\d+)\]$/.exec(part);
    if (!match) {
      if (part) para.appendChild(document.createTextNode(part));
      continue;
    }
    const index = Number(match[1]);
    const resolved = payload.sources.some((s) => s.index === index);
    const chip = el("button", resolved ? "cite" : "cite broken", String(index));
    chip.type = "button";
    chip.dataset.index = String(index);
    if (resolved) {
      chip.title = `Source ${index}`;
      chip.setAttribute("aria-label", `Jump to source ${index}`);
    } else {
      chip.title = `Citation ${index} does not resolve to a retrieved source`;
      chip.setAttribute("aria-label", chip.title);
    }
    chip.addEventListener("click", () => focusSource(index));
    para.appendChild(chip);
  }

  box.appendChild(para);

  $("demo-badge").hidden = !payload.is_demo;

  const broken = payload.dangling_citations || [];
  $("dangling").hidden = broken.length === 0;
  if (broken.length) {
    $("dangling").textContent =
      `The answer cites ${broken.join(", ")}, which was not retrieved. ` +
      "Treat those claims as unsupported.";
  }

  $("cited-badge").hidden = !payload.sources.length;
  $("cited-badge").textContent = `cited ${Math.round((payload.coverage || 0) * 100)}% of sources`;
}

/* ── sources ────────────────────────────────────────────────────────────── */

function renderSources(payload) {
  const list = $("sources");
  list.replaceChildren();
  state.sources = payload.sources;
  $("corpus-note").textContent = `${payload.sources.length} retrieved`;

  if (!payload.sources.length) {
    const empty = el("div", "empty");
    empty.append(
      el("strong", null, "Nothing matched."),
      document.createTextNode(
        " The gateway will not answer from memory when a question has no " +
          "supporting passage — try different terms, or raise the passage count."
      )
    );
    list.appendChild(empty);
    return;
  }

  payload.sources.forEach((s, position) => {
    const row = el("li", s.cited ? "used" : null);
    row.dataset.index = String(s.index);

    const head = el("div", "ledger-head");
    const id = el("span", "ledger-id", s.doc_id);
    id.dataset.n = String(s.index);
    head.append(id, el("span", "score", s.score.toFixed(2)));
    row.append(head, el("p", "ledger-text", s.excerpt));

    row.addEventListener("click", () => lightSources(s.index));
    row.addEventListener("mouseenter", () => lightChips(s.index));
    row.addEventListener("mouseleave", () => lightChips(null));
    list.appendChild(row);

    // The one authored moment: rows settle into place in the order the answer
    // cites them, so the path from a claim to its evidence is visible as motion.
    if (!reducedMotion()) {
      row.classList.add("settle");
      row.style.animationDelay = `${position * 34}ms`;
    }
  });
}

/* ── instrumentation ────────────────────────────────────────────────────── */

function renderUsage(payload) {
  const u = payload.usage || {};
  const rows = [
    ["Model", u.model || "—"],
    ["Retrieval", `${payload.retrieval_ms} ms`],
    ["Generation", `${payload.generation_ms} ms`],
    ["Prompt tokens", u.prompt_tokens ?? 0],
    ["Completion tokens", u.completion_tokens ?? 0],
    ["Total tokens", u.total_tokens ?? 0],
  ];
  fillFigures($("usage"), rows);

  fillFigures($("mode"), [
    ["Retriever", payload.retrieval_mode || "—"],
    ["Verifying", payload.verified ? "yes" : "no"],
    ["Dangling citations", (payload.dangling_citations || []).length],
  ]);
}

function fillFigures(dl, rows) {
  dl.replaceChildren();
  for (const [key, value] of rows) {
    dl.append(el("dt", null, key), el("dd", null, String(value)));
  }
}

function renderAudit(payload) {
  const box = $("audit");
  const audit = payload.audit;

  if (!audit || !audit.verified) {
    box.hidden = true;
    return;
  }

  box.hidden = false;
  const parts = [
    `${audit.supported} of ${audit.claims.length} claims grounded in the sources`,
  ];
  if (audit.abstained) parts.push(`answer withheld — ${payload.abstain_reason.replace(/_/g, " ")}`);
  if (audit.revisions) parts.push(`${audit.revisions} revision`);
  $("audit-text").textContent = parts.join(" · ");

  /* Grounding is not relevance. Each sentence here really does appear in the
     retrieved passage, and the strip would still read "verified" on an answer to a
     question the corpus cannot address — ask about Kubernetes ingress and the
     verifier happily grounds three sentences from the Docker chapter.

     Lexical overlap was measured as a relevance gate and rejected: across the 51
     in-corpus questions the overlap ranged 0.167-0.667, and off-topic questions
     scored up to 0.500 — higher than the in-corpus median. A threshold there would
     abstain on good questions and answer bad ones.

     So the limit is stated where it is read, rather than papered over with a gate that
     does not work. */
  const scope = $("audit-scope");
  scope.hidden = false;
  scope.textContent =
    "Checks that each sentence appears in the sources. It does not check that the sources answer your question.";

  box.dataset.tone = payload.abstained ? "bad" : audit.unsupported ? "warn" : "ok";
}

/* ── claims ─────────────────────────────────────────────────────────────── */

/* Per-claim attribution exists to make a *partial* answer visible. A single
   answer-level verdict hides the case that matters: right on one point, invented on
   another. */
function renderClaims(payload) {
  const block = $("claims-block");
  const list = $("claims");
  list.replaceChildren();

  const claims = payload.claims || [];
  block.hidden = claims.length === 0;
  if (!claims.length) return;

  const verdicts = new Map((payload.audit?.claims || []).map((c) => [c.text, c]));

  for (const claim of claims) {
    const verdict = verdicts.get(claim.text);
    const flagged = claim.uncited || (verdict && !verdict.supported);
    const refs = claim.citations?.length ? claim.citations.join(", ") : "no citation";
    const label = verdict
      ? `${verdict.supported ? "supported" : "unsupported"} — ${claim.text} [${refs}]`
      : `${claim.text} [${refs}]`;
    list.appendChild(el("li", flagged ? "flagged" : null, label));
  }
}

/* ── linking a claim to its evidence ────────────────────────────────────── */

function lightSources(index) {
  document.querySelectorAll(".sources li").forEach((row) => {
    row.classList.toggle("lit", row.dataset.index === String(index));
  });
  const row = document.querySelector(`.sources li[data-index="${index}"]`);
  if (row) row.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "nearest" });
}

function lightChips(index) {
  document.querySelectorAll(".cite").forEach((chip) => {
    chip.classList.toggle("lit", index !== null && chip.dataset.index === String(index));
  });
}

function focusSource(index) {
  lightChips(index);
  lightSources(index);
}

/* ── states ─────────────────────────────────────────────────────────────── */

function setStatus(message, tone) {
  const node = $("status");
  if (!message) {
    node.hidden = true;
    return;
  }
  node.hidden = false;
  node.className = tone === "bad" ? "alert alert-bad" : "alert";
  node.textContent = message;
}

function reset() {
  $("results").hidden = true;
  $("answer-wrap").hidden = true;
  $("sources-panel").hidden = true;
  $("rail").hidden = true;
  $("claims-block").hidden = true;
  $("audit").hidden = true;
  $("dangling").hidden = true;
  setStatus(null);
}

async function ask() {
  const question = $("question").value.trim();
  if (!question) {
    setStatus("Ask something first — the gateway answers from retrieved passages, not memory.", "bad");
    $("question").focus();
    return;
  }

  const button = $("go");
  button.disabled = true;
  button.textContent = "Asking";
  reset();
  setStatus("Retrieving passages, then checking every claim against them.");

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question, top_k: Number($("topk").value) || 5 }),
    });

    const data = await res.json();

    if (!res.ok) {
      // Surface the per-provider reason. A bare "request failed" makes a
      // misconfiguration indistinguishable from a model failure.
      const detail = data.providers
        ? Object.entries(data.providers).map(([name, why]) => `${name}: ${why}`).join(" · ")
        : data.detail || data.error || "unknown error";
      setStatus(`${res.status} — ${detail}`, "bad");
      return;
    }

    $("results").hidden = false;
    $("answer-wrap").hidden = false;
    $("sources-panel").hidden = false;
    $("rail").hidden = false;

    renderAnswer(data);
    renderSources(data);
    renderUsage(data);
    renderAudit(data);
    renderClaims(data);
    setStatus(null);

    if (!reducedMotion()) {
      const answer = $("answer-wrap");
      answer.classList.add("settle");
      answer.addEventListener("animationend", () => answer.classList.remove("settle"), { once: true });
    }
  } catch (err) {
    setStatus(`Could not reach the service — ${err.message}`, "bad");
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

/* ── wiring ─────────────────────────────────────────────────────────────── */

function init() {
  const list = $("examples");
  for (const text of EXAMPLES) {
    const button = el("button", null, text);
    button.type = "button";
    button.addEventListener("click", () => {
      $("question").value = text;
      $("question").focus();
    });
    list.appendChild(button);
  }

  $("go").addEventListener("click", ask);

  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
      event.preventDefault();
      ask();
    }
  });

  refreshHealth();
  setInterval(refreshHealth, 20000);
}

document.addEventListener("DOMContentLoaded", init);
