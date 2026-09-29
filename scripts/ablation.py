#!/usr/bin/env python3
"""Run the ablation. Thin entry point; the harness lives in gateway.rag.ablation."""

import sys

from gateway.rag.ablation import main

if __name__ == "__main__":
    sys.exit(main())
