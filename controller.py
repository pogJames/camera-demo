"""Guided-sequence demo state machine (pure logic, no I/O). See CLAUDE.md."""

import threading


class DemoController:
    def __init__(self, registers, idle_state, scan_state, confirm_frames, regress_frames):
        self.registers = list(registers)
        self.idle_state = idle_state
        self.scan_state = scan_state
        self.confirm_frames = max(1, int(confirm_frames))
        self.regress_frames = max(1, int(regress_frames))
        self._validate(idle_state, scan_state)
        self._lock = threading.Lock()
        self._reset_locked()

    def _validate(self, *patterns):
        n = len(self.registers)
        for state in patterns:
            if len(state) != n or set(state) - set("01"):
                raise ValueError(f"lamp pattern error: {state!r}")

    def _reset_locked(self):
        self.steps = []
        self.labels = []
        self._index = 0
        self._error = False
        self._error_got = None
        self._streak = {}
        self._miss = {}
        self._outside = {}
        self._visible = set()
        self._scan = None

    def reset(self):
        with self._lock:
            self._reset_locked()

    def load(self, code, fmt=None, specs=None):
        """Install a scanned product's step list. No steps -> nothing to run."""
        steps = list((specs or {}).get("steps", ()))
        self._validate(*[s["state"] for s in steps])
        with self._lock:
            self._reset_locked()
            self.steps = steps
            self.labels = list(dict.fromkeys(s["label"] for s in steps))
            self._scan = {"code": code, "format": fmt, "specs": specs}

    def update(self, labels, visible=None):
        with self._lock:
            if not self.steps:            # no product scanned yet
                return self._state_locked(), self._regs_locked()
            self._visible = set(labels if visible is None else visible)
            present = self._confirmed_locked(labels)
            expected = self._expected_locked()
            completed = {s["label"] for s in self.steps[:self._index]}
            wrong = (present - {expected} - completed) if self._index else set()

            if expected in present:
                self._error = False
                self._error_got = None
                self._index += 1
            elif wrong:
                self._error = True
                self._error_got = sorted(wrong)[0]
            else:
                self._error = False
                self._error_got = None
                self._regress_locked(present)

            return self._state_locked(), self._regs_locked()

    def snapshot(self):
        with self._lock:
            return self._state_locked(), self._regs_locked()

    def _confirmed_locked(self, labels):
        for lab in self.labels:
            seen = lab in labels # check if label is seen in the current frame
            self._streak[lab] = self._streak.get(lab, 0) + 1 if seen else 0
            self._miss[lab] = 0 if seen else self._miss.get(lab, 0) + 1
            outside = not seen and lab in self._visible
            self._outside[lab] = self._outside.get(lab, 0) + 1 if outside else 0
        return self._present_locked()

    def _expected_locked(self):
        if self._index >= len(self.steps):
            return None
        return self.steps[self._index]["label"]

    def _regress_locked(self, present):
        if self._index == 0:
            return
        last = self.steps[self._index - 1]
        if last.get("container") not in present:  # no container in view, no evidence
            return
        if self._expected_locked() in self._visible:  # next item mid-placement
            return
        if self._miss.get(last["label"], 0) >= self.regress_frames:
            self._index -= 1

    def _state_locked(self):
        steps = []
        for i, s in enumerate(self.steps):
            state = "done" if i < self._index else ("active" if i == self._index else "pending")
            steps.append({"title": s["title"], "label": s["label"], "state": state})
        return {
            "steps": steps,
            "current": self._index,
            "loaded": bool(self.steps),
            "complete": bool(self.steps) and self._index >= len(self.steps),
            "error": {"active": self._error,
                      "expected": self._expected_locked(),
                      "got": self._error_got},
            "misplaced": self._misplaced_locked(),
            "scan": self._scan,
            "lamps": [{"name": f"L{i + 1}", "on": bit == "1"}
                      for i, bit in enumerate(self._pattern_locked())],
        }

    def _misplaced_locked(self):
        if self._error or not self.steps or self._index >= len(self.steps):
            return None
        step = self.steps[self._index]
        if not step.get("container"):  # a box state is never "inside" anything
            return None
        outside = self._outside.get(step["label"], 0) >= self.confirm_frames
        return step["label"] if outside else None

    def _present_locked(self):
        return {lab for lab, n in self._streak.items() if n >= self.confirm_frames}

    def _pattern_locked(self):
        if self._index == 0:
            return self.scan_state if self.steps else self.idle_state
        return self.steps[self._index - 1]["state"]

    def _regs_locked(self):
        return {addr: bit == "1"
                for addr, bit in zip(self.registers, self._pattern_locked())}
