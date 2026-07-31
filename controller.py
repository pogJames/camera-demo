"""Guided-sequence demo state machine (pure logic, no I/O). See CLAUDE.md."""

import threading
import time


class DemoController:
    def __init__(self, steps, confirm_frames, auto_reset_secs,
                 step_regs, fault_reg):
        self.steps = list(steps)
        self.confirm_frames = max(1, int(confirm_frames))
        self.auto_reset_secs = auto_reset_secs
        self.step_regs = list(step_regs)
        self.fault_reg = fault_reg
        self._lock = threading.Lock()
        self._reset_locked()

    def _reset_locked(self):
        self._index = 0
        self._fault = False
        self._fault_got = None
        self._streak = {}
        self._done_at = None

    def reset(self):
        with self._lock:
            self._reset_locked()

    def update(self, labels, now=None):
        now = time.time() if now is None else now
        with self._lock:
            present = self._confirmed_locked(labels)
            n = len(self.steps)

            if self._index >= n:
                if self._done_at is None:
                    self._done_at = now
                elif self.auto_reset_secs and now - self._done_at >= self.auto_reset_secs:
                    self._reset_locked()
            else:
                expected = self.steps[self._index]
                completed = set(self.steps[:self._index])
                wrong = present - {expected} - completed
                if wrong:
                    self._fault = True
                    self._fault_got = sorted(wrong)[0]
                elif expected in present:
                    self._index += 1
                    self._fault = False
                    self._fault_got = None
                    if self._index >= n:
                        self._done_at = now
                else:
                    self._fault = False
                    self._fault_got = None

            return self._state_locked(), self._regs_locked()

    def snapshot(self):
        with self._lock:
            return self._state_locked(), self._regs_locked()

    def _confirmed_locked(self, labels):
        present = set()
        for lab in self.steps:
            self._streak[lab] = self._streak.get(lab, 0) + 1 if lab in labels else 0
            if self._streak[lab] >= self.confirm_frames:
                present.add(lab)
        return present

    def _state_locked(self):
        n = len(self.steps)
        steps = []
        for i, lab in enumerate(self.steps):
            state = "done" if i < self._index else ("active" if i == self._index else "pending")
            steps.append({"label": lab, "state": state})
        expected = self.steps[self._index] if self._index < n else None
        return {
            "steps": steps,
            "current": self._index,
            "complete": self._index >= n,
            "fault": {"active": self._fault, "expected": expected, "got": self._fault_got},
        }

    def _regs_locked(self):
        regs = {addr: i < self._index for i, addr in enumerate(self.step_regs)}
        if self.fault_reg is not None:
            regs[self.fault_reg] = self._fault
        return regs
