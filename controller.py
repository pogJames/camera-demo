"""Guided-sequence demo state machine (pure logic, no I/O). See CLAUDE.md."""

import threading


class DemoController:
    def __init__(self, steps, registers, idle_state, confirm_frames, regress_frames):
        self.steps = list(steps)
        self.registers = list(registers)
        self.idle_state = idle_state
        self.confirm_frames = max(1, int(confirm_frames))
        self.regress_frames = max(1, int(regress_frames))
        self.labels = list(dict.fromkeys(s["label"] for s in self.steps))
        self._validate()
        self._lock = threading.Lock()
        self._reset_locked()

    def _validate(self):
        n = len(self.registers)
        for state in [self.idle_state] + [s["state"] for s in self.steps]:
            if len(state) != n or set(state) - set("01"):
                raise ValueError(
                    f"lamp pattern error: {state!r}")

    def _reset_locked(self):
        self._index = 0
        self._error = False
        self._error_got = None
        self._streak = {}
        self._miss = {}
        self._visible = set()

    def reset(self):
        with self._lock:
            self._reset_locked()

    def update(self, labels, visible=None):
        with self._lock:
            self._visible = set(labels if visible is None else visible)
            present = self._confirmed_locked(labels)
            expected = self._expected_locked()
            completed = {s["label"] for s in self.steps[:self._index]}
            wrong = present - {expected} - completed

            if wrong:
                self._error = True
                self._error_got = sorted(wrong)[0]
            else:
                self._error = False
                self._error_got = None
                if expected in present:
                    self._index += 1
                else:
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
            "complete": self._index >= len(self.steps),
            "error": {"active": self._error,
                      "expected": self._expected_locked(),
                      "got": self._error_got},
            "misplaced": self._misplaced_locked(),
            "lamps": [{"name": f"L{i + 1}", "on": bit == "1"}
                      for i, bit in enumerate(self._pattern_locked())],
        }

    def _misplaced_locked(self):
        expected = self._expected_locked()
        if expected is None or self._error or expected in self._present_locked():
            return None
        return expected if expected in self._visible else None

    def _present_locked(self):
        return {lab for lab, n in self._streak.items() if n >= self.confirm_frames}

    def _pattern_locked(self):
        if self._index == 0:
            return self.idle_state
        return self.steps[self._index - 1]["state"]

    def _regs_locked(self):
        return {addr: bit == "1"
                for addr, bit in zip(self.registers, self._pattern_locked())}
