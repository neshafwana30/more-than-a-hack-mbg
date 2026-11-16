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
    set_piece: np.ndarray      # [0, 1] soft evidence for a potential restart
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


def _normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def _team_code(value: str, team_map: dict[str, str]) -> int | None:
    side = team_map.get(_normalize(value))
    if side == "H":
        return 0
    if side == "A":
        return 1
    return None


def tactical_evidence(events: list[dict], n_frames: int, cfg: dict) -> TacticalEvidence:
    if n_frames < 0:
        raise ValueError("n_frames must be nonnegative")

    weights = {str(key).strip().upper(): float(val)
               for key, val in cfg["event_weights"].items()}
    configured_team_map = cfg.get("team_map", {
        "Barcelona": "H",
        "Atletic Club": "A",
    })
    team_map = {_normalize(str(team)): str(side).strip().upper()
                for team, side in configured_team_map.items()}
    if set(team_map.values()) != {"H", "A"}:
        raise ValueError("tactical.team_map must map team names to both H and A")

    taper_frames = int(cfg.get("event_edge_taper_frames", 0))
    if taper_frames < 0:
        raise ValueError("event_edge_taper_frames must be nonnegative")

    priors = np.zeros(n_frames, dtype=np.float32)
    set_piece = np.zeros(n_frames, dtype=np.float32)
    for event in events:
        start = max(0, min(n_frames, int(event["start"])))
        end = max(start, min(n_frames, int(event["end"])))
        if start == end:
            continue
        code = str(event["code"]).strip().upper()
        duration = end - start
        taper = min(taper_frames, duration // 2)
        if taper:
            positions = np.arange(duration, dtype=np.float32)
            envelope = np.minimum(
                1.0,
                np.minimum((positions + 1) / taper, (duration - positions) / taper),
            )
        else:
            envelope = 1.0

        if code == "SET PIECES":
            set_piece[start:end] = np.maximum(set_piece[start:end], envelope)

        team = _team_code(event["tags"].get("team", ""), team_map)
        if team is None or code not in weights:
            continue
        value = weights[code] * (1 if team == 0 else -1)
        priors[start:end] += value * envelope

    priors = np.clip(priors * float(cfg.get("strength", 1.0)), -3.0, 3.0)
    return TacticalEvidence(priors, set_piece, len(events))
