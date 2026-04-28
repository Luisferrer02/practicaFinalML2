"""Atajo para reindexar todo el corpus desde cero. Equivalente a:
    python -m src.ingest --reset
"""
from src.ingest import main

if __name__ == "__main__":
    main(reset=True)
