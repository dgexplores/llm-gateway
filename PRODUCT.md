# Gateway — product surface

The product surface for a retrieval gateway whose entire premise is that you can check
its work. That premise drives every decision below: if the interface hides how an answer
was produced, the product has failed regardless of how good the answer was.

## Who it is for

A developer evaluating whether to route their LLM traffic through this gateway, and an
engineer debugging an answer that came out wrong. The first visit is a judgement; the
second visit is a diagnosis. Both need the evidence, and neither needs to be an expert
in information retrieval to read it.

## The one thing

Type a question, get an answer, and see every claim traced to a specific retrieved
source — with the retrieval scores, the provider chain, the token cost, and the
verification verdict for that exact text.

## Direction

**Calm product surface.** Light, warm, generous whitespace, soft surfaces, quiet
typography. Deliberately not a dark "AI console" and not a dense monitoring dashboard.

The reasoning: the default for anything labelled an LLM tool is a dark terminal
aesthetic. That reads as impressive and scores well in a screenshot, but it is hostile
to the reading task this product is actually about. Retrieval evidence is paragraphs of
prose with scores attached. Prose is read, not scanned. A calm, light, document-like
surface supports reading; a dark terminal surface fights it.

Restraint carries the weight. One authored settle animation on the answer, and nothing
else moves. The interface is not trying to look alive.

## Rules this surface holds to

- **Evidence is not optional.** Scores, cost, provider, and verification stay on screen.
  Nothing collapses behind a hover.
- **Say what the mode is.** Demo output and live generation are labelled differently, at
  the point of use. A reader must never mistake one for the other.
- **Prose is a first-class element.** The answer is set as reading text with comfortable
  measure. Instrumentation is a separate rail, not interleaved into the prose.
- **States are explicit.** Loading, empty, error, and abstention each have a distinct
  appearance. An empty result is never an empty box.
- **Keyboard and motion are respected.** Citation chips are real buttons; focus is
  visible; `prefers-reduced-motion` removes the animation entirely.
- **Works offline.** No web fonts, no CDN, no external requests. The favicon is inlined.

## What a reviewer should feel

That this was built by someone who has read the same research papers, and who noticed
which parts of their output were wrong. The interface is an argument for trusting the
tool, made out of showing the work rather than claiming competence.
