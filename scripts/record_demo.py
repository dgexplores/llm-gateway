#!/usr/bin/env python3
"""Record a scripted browser session of the UI as an animated GIF.

A screenshot proves the interface looks right at one size on one day. It does not show a
reader what the thing does, which for a gateway is the part that matters: a question goes
in, passages get retrieved and cited, an answer comes out, and the page tells you how
grounded it is.

So this drives a real browser against the real server — no stubbed responses, no
pre-baked frames — and captures the viewport repeatedly while the page settles, then
assembles the frames into a GIF with ffmpeg.

The scenario deliberately includes a question the corpus cannot answer. The interesting
moment in a tool like this is not the happy path, it is the moment it declines, so the
recording would be dishonest if every frame were a success.

Requires playwright and ffmpeg. Screenshots land in docs/demo/frames/ and the result in
docs/demo/gateway-demo.gif.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "docs" / "demo" / "frames"
OUT = ROOT / "docs" / "demo" / "gateway-demo.gif"

WIDTH, HEIGHT = 1280, 860

# (question, note). The second question is out of corpus on purpose.
SCENARIO: list[tuple[str, str]] = [
    ("How do I add a query parameter to a path operation?", "answerable"),
    ("Which employee invented the pneumatic stapler?", "out of corpus"),
]


def require(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found. Install it, or skip the demo.")


def capture(page, label: str, index: int, settle_ms: int = 450) -> None:
    """One frame: let the page settle, then screenshot into the frame directory."""
    page.wait_for_timeout(settle_ms)
    path = FRAMES / f"{index:03d}-{label}.png"
    page.screenshot(path=str(path))
    print(f"  {path.name}")


def main() -> None:
    require("ffmpeg")
    base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    FRAMES.mkdir(parents=True, exist_ok=True)
    for stale in FRAMES.glob("*.png"):
        stale.unlink()

    index = 0
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
        print(f"driving {base}")

        page.goto(base, wait_until="networkidle")
        capture(page, "idle", index := index + 1)

        for question, kind in SCENARIO:
            box = page.get_by_role("textbox").first
            box.click()
            box.fill(question)

            capture(page, "typed", index := index + 1, settle_ms=350)

            # The provider quotes retrieved passages, so answers are not instant.
            box.press("Enter")
            page.wait_for_timeout(1200)
            capture(page, f"{kind}-answered", index := index + 1, settle_ms=1400)

        browser.close()

    frames = sorted(FRAMES.glob("*.png"))
    if len(frames) < 3:
        sys.exit(f"only {len(frames)} frame(s) captured; is the server running?")

    # A slow, even cadence. A GIF of a UI that flickers reads as a bug, so each frame gets
    # equal weight rather than pacing frames to whatever delay the page happened to take.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-framerate", "1/2",
        "-pattern_type", "glob", "-i", str(FRAMES / "*.png"),
        "-vf",
        # Split into a palette pass and an apply pass: gif is palettised, and a plain
        # scale+dither filter chain quantises per-frame instead.
        f"scale={WIDTH}:-1:flags=lanczos,split[a][b];"
        "[a]palettegen=max_colors=128[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=4",
        "-loop", "0", str(OUT),
    ]
    subprocess.run(cmd, check=True)
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT.relative_to(ROOT)} ({size_mb:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()