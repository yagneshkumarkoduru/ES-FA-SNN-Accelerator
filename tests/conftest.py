"""Shared path setup for the ES-FA test suite (torch-free, numpy-only)."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for sub in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "implementations", "v3_csharp_net9_hal_sd_flashattention"),
):
    if sub not in sys.path:
        sys.path.insert(0, sub)
