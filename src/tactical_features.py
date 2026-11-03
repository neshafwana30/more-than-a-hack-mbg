"""Convert automatically generated SportsCode phases to *soft* team priors.

No human-produced events from this match are read. 'N/A' team tags are ignored.
Event intervals are NOT copied as possession labels.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class TacticalEvidence:
    log_odds_home: np.ndarray  # positive -> H, negative -> A
    set_piece: np.ndarray      # 0/1 conservative evidence for potential restart
    event_count: int


def read_events(xml_path: Path, fps: float) -> list[dict]:
    if not Path(xml_path).is_file():
        raise FileNotFoundError(f"Missing Smart Tagging: {xml_path}")
    events = []
    for node in ET.parse(xml_path).iter("instance"):
        tags = {label.findtext("group", default="").strip().lower():
                label.findtext("text", default="").strip()
                for label in node.findall("label")}
        events.append({
            "code": (node.findtext("code") or "").strip().upper(),
            "start": int(round(float(node.findtext("start")) * fps)),
            "end": int(round(float(node.findtext("end")) * fps)),
            "tags": tags,
        })
    return events


def _team_code(value: str) -> int | None:
    value = value.strip().casefold()
    if "barcelona" in value:
        return 0
    if ("atletic" in value or "athletic" in value) and "club" in value:
        return 1
    return None


def tactical_evidence(events: list[dict], n_frames: int, cfg: dict) -> TacticalEvidence:
    weights = {str(key).upper(): float(val) for key, val in cfg["event_weights"].items()}
    delta = np.zeros(n_frames + 1, dtype=np.float32)
    piece_delta = np.zeros(n_frames + 1, dtype=np.int32)
    for event in events:
        start = max(0, min(n_frames, int(event["start"])))
        end = max(start, min(n_frames, int(event["end"])))
        if start == end:
            continue
        code = event["code"]
        if code == "SET PIECES":
            piece_delta[start] += 1
            piece_delta[end] -= 1
        team = _team_code(event["tags"].get("team", ""))
        if team is None or code not in weights:
            continue
        value = weights[code] * (1 if team == 0 else -1)
        delta[start] += value
        delta[end] -= value
    priors = np.cumsum(delta[:-1], dtype=np.float32)
    priors = np.clip(priors * float(cfg.get("strength", 1.0)), -3.0, 3.0)
    return TacticalEvidence(priors, np.cumsum(piece_delta[:-1]) > 0, len(events))
