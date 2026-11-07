"""RA-TBPS: retrieval-augmented test-time adaptation for text-based person search.

Pipeline (see README.md):
  extract_aptm.py  frozen APTM (X-VLM) -> cached features per split (.npz)
  memory.py        evidence memory: labelled source pairs + bidirectionally
                   agreed pseudo-pairs from the unlabelled target split
  model.py         identity-aligned retriever R and context-grounded scorer g
  run.py           train R and g on the source cache, evaluate on target caches
"""
