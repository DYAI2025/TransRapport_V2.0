import json
from pathlib import Path
from typing import Dict, Any, List


class IntuitionTelemetry:
    def __init__(self, bundle_id: str, data_dir: Path, scoring_cfg: Dict[str, Any]):
        self.bundle_id = bundle_id
        self.data_dir = data_dir
        self.cfg = scoring_cfg or {}
        self.multiplier = float(self.cfg.get("intuition_multiplier_on_confirm", 1.5))
        self.win_confirm = int(self.cfg.get("intuition_confirm_window_messages", 6))
        self.win_decay = int(self.cfg.get("intuition_decay_after_messages", 12))
        self.states: Dict[str, Dict[str, Any]] = {}

    def _ensure(self, name: str):
        if name not in self.states:
            self.states[name] = {
                "name": name,
                "provisional_hits": 0,
                "confirmed_hits": 0,
                "last_provisional_at_msg_idx": None,
                "last_confirmed_at_msg_idx": None,
                "decay_pending_after_msg_idx": None,
                "multiplier_active": False,
                "multiplier": self.multiplier,
                "window_confirm_messages": self.win_confirm,
                "window_decay_messages": self.win_decay,
            }

    def on_provisional(self, name: str, msg_idx: int):
        self._ensure(name)
        st = self.states[name]
        st["provisional_hits"] += 1
        st["last_provisional_at_msg_idx"] = msg_idx
        # confirm if within window of last provisional
        last = st.get("last_confirmed_at_msg_idx")
        # rule: X-of-Y semantics are handled by marker engine; here we simply watch for proximity of provisional hits
        if st["last_provisional_at_msg_idx"] is not None:
            if st["provisional_hits"] >= 2 and (msg_idx - (last or msg_idx)) <= self.win_confirm:
                self.on_confirm(name, msg_idx)

    def on_confirm(self, name: str, msg_idx: int):
        self._ensure(name)
        st = self.states[name]
        st["confirmed_hits"] += 1
        st["last_confirmed_at_msg_idx"] = msg_idx
        st["multiplier_active"] = True
        st["decay_pending_after_msg_idx"] = msg_idx + self.win_decay

    def tick(self, msg_idx: int):
        for st in self.states.values():
            if st.get("multiplier_active") and st.get("decay_pending_after_msg_idx") is not None:
                if msg_idx >= st["decay_pending_after_msg_idx"]:
                    st["multiplier_active"] = False

    def to_json(self, session_id: str) -> Dict[str, Any]:
        return {
            "session_id": session_id,
            "bundle_id": self.bundle_id,
            "intuition_states": list(self.states.values()),
        }

    def save(self, session_id: str):
        obj = self.to_json(session_id)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "telemetry_intuitions.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

