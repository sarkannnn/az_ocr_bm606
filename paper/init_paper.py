import os
import sys
from pathlib import Path

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding="utf-8")

paper_fig_dir = Path("paper/figures")
paper_fig_dir.mkdir(parents=True, exist_ok=True)
paper_tbl_dir = Path("paper/tables")
paper_tbl_dir.mkdir(parents=True, exist_ok=True)

print(f"Figures directory initialized at {paper_fig_dir.resolve()}")
print(f"Tables directory initialized at {paper_tbl_dir.resolve()}")
