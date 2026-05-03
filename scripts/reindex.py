"""Atajo para reindexar todo el corpus desde cero. Equivalente a:
    python -m src.ingest --reset
"""
from src.ingest import run_ingest

if __name__ == "__main__":
    run_ingest(reset=True)
