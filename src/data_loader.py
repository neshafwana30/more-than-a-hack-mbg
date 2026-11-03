"""Stream 1-based tactical FIFA/EPTS tracking and its XML metadata.

ATD rows look like ``1:x,y,s;...:ball_x,ball_y,ball_s``. Slot identities are
tracklets, NOT player identities. Slot-to-team membership is in the metadata.
Only the tactical recording is used; broadcast video is not time-synchronized.
"""
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class Tracking:
    players_xy: np.ndarray       # (frames, slots, 2), normalized [0..1]; NaN if absent
    ball_xy: np.ndarray          # (frames, 2), normalized coordinates; NaN if absent
    ball_speed: np.ndarray       # speed provided in ATD (m/s), NaN if missing
    teams: np.ndarray            # (slots,), 0=H, 1=A
    fps: int
    pitch: tuple[float, float]

    @property
    def n_frames(self) -> int:
        return self.ball_xy.shape[0]

    @property
    def player_count(self) -> np.ndarray:
        return np.isfinite(self.players_xy).all(axis=2).sum(axis=1)


def read_teams(meta_path: Path) -> np.ndarray:
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    if len(team_ids) != 2:
        raise ValueError("ATD metadata must contain exactly two teams")
    player_team_ids = [p.get("teamId") for p in root.iter("Player")]
    if not player_team_ids:
        raise ValueError("ATD metadata contains no player track slots")
    unknown = set(player_team_ids).difference(team_ids)
    if unknown:
        raise ValueError(f"Unknown team IDs assigned to slots: {unknown}")
    return np.asarray([0 if v == team_ids[0] else 1 for v in player_team_ids],
                      dtype=np.int8)


def load_tracking(meta_path: Path, raw_path: Path, n_frames: int, fps: int,
                  pitch: tuple[float, float]) -> Tracking:
    """Read exactly n_frames rows, rejecting missing/duplicate/out-of-order IDs.

    This deliberately never uses the public possession ground truth.
    """
    meta_path, raw_path = Path(meta_path), Path(raw_path)
    for path in (meta_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}. Obtain the full licensed tactical ATD. "
                                    "The provided ZIP includes only *_sample.txt.")
    teams = read_teams(meta_path)
    n_slots = len(teams)
    players_xy = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball_speed = np.full(n_frames, np.nan, dtype=np.float32)
    with raw_path.open(encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            if index >= n_frames:
                raise ValueError(f"ATD file has more than {n_frames} records")
            head, sep, payload = line.partition(":")
            if not sep:
                raise ValueError(f"Malformed ATD row at line {index+1}")
            try:
                frame_id = int(head)
            except ValueError as exc:
                raise ValueError(f"Invalid ATD frame ID at row {index+1}") from exc
            if frame_id != index + 1:
                raise ValueError(f"ATD frame IDs must be sequential 1..N: "
                                 f"expected {index+1}, observed {frame_id}")
            # Following frame id are n_slots triples and one ball triple.
            # np.fromstring treats the literal 'NaN' as missing; we retain it.
            row = np.fromstring(payload.replace(";", ",").replace(":", ","),
                                dtype=np.float32, sep=",")
            expected = (n_slots + 1) * 3
            if row.size != expected:
                raise ValueError(f"ATD frame {frame_id}: expected {expected} "
                                 f"numbers, found {row.size}")
            triples = row.reshape(n_slots + 1, 3)
            players_xy[index] = triples[:-1, :2]
            ball_xy[index] = triples[-1, :2]
            ball_speed[index] = triples[-1, 2]
        else:
            read_count = index + 1 if 'index' in locals() else 0
    if read_count != n_frames:
        raise ValueError(f"Incomplete ATD: read {read_count} rows; expected "
                         f"{n_frames}. The ZIP contains a 10-frame sample only; "
                         "obtain BAR-ATH_tactical_atd.txt from the organizer.")
    return Tracking(players_xy, ball_xy, ball_speed, teams, fps, pitch)


def input_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    tactical = Path(data_dir) / "tactical"
    return (tactical / "BAR-ATH_tactical_atd.xml",
            tactical / "BAR-ATH_tactical_atd.txt",
            tactical / "BAR-ATH_tactical_smart_tagging.xml")
