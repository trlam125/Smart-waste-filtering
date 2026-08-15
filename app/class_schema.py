from __future__ import annotations

# Canonical class order shared by training, checkpoints, inference and feedback vectors.
WASTE_CLASS_KEYS: tuple[str, ...] = (
    "plastic_rigid",
    "plastic_film",
    "paper",
    "cardboard",
    "metal",
    "glass",
    "organic",
    "hazardous",
    "electronic",
    "textile",
    "other",
)
