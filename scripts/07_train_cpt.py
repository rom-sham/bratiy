"""Run from any working directory; paths resolve against the project root."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cli import main

if __name__ == "__main__":
    main("train_cpt")
