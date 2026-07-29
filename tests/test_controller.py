"""Pure-logic tests for the demo state machine (no hardware). See CLAUDE.md."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from controller import DemoController

STEPS = ["bottle", "cell phone", "scissors"]
REGS = [0x000D, 0x000E, 0x000F]
FAULT = 0x0010


def make(confirm_frames=1, auto_reset_secs=5):
    return DemoController(STEPS, confirm_frames, auto_reset_secs, REGS, FAULT)


def test_advances_in_order():
    c = make()
    for i, label in enumerate(STEPS):
        state, regs = c.update({label})
        assert state["current"] == i + 1
        assert regs[REGS[i]] is True
        assert regs[FAULT] is False
    assert c.update(set())[0]["complete"] is True


def test_wrong_part_faults_and_freezes():
    c = make()
    state, regs = c.update({"scissors"})   # expected bottle
    assert state["fault"]["active"] is True
    assert state["fault"]["got"] == "scissors"
    assert state["current"] == 0            # frozen, did not advance
    assert regs[FAULT] is True


def test_fault_auto_clears_when_removed():
    c = make()
    assert c.update({"scissors"})[0]["fault"]["active"] is True
    state, regs = c.update(set())          # wrong part gone
    assert state["fault"]["active"] is False
    assert regs[FAULT] is False
    assert c.update({"bottle"})[0]["current"] == 1   # resumes


def test_completed_step_is_not_a_fault():
    c = make()
    c.update({"bottle"})                     # step 0 done, now expect phone
    state = c.update({"bottle", "cell phone"})[0]   # old part still visible
    assert state["fault"]["active"] is False
    assert state["current"] == 2


def test_debounce_needs_consecutive_frames():
    c = make(confirm_frames=2)
    assert c.update({"bottle"})[0]["current"] == 0   # 1st sighting: not yet
    assert c.update({"bottle"})[0]["current"] == 1   # 2nd: confirmed
    # a gap resets the streak
    c2 = make(confirm_frames=2)
    c2.update({"bottle"})
    c2.update(set())
    assert c2.update({"bottle"})[0]["current"] == 0


def test_auto_reset_after_completion():
    c = make(auto_reset_secs=5)
    c.update({"bottle"}, now=100)
    c.update({"cell phone"}, now=100)
    state = c.update({"scissors"}, now=100)[0]
    assert state["complete"] is True
    assert c.update(set(), now=104)[0]["complete"] is True     # not yet
    assert c.update(set(), now=106)[0]["current"] == 0         # reset fired


def test_manual_reset():
    c = make()
    c.update({"bottle"})
    c.reset()
    state, regs = c.snapshot()
    assert state["current"] == 0
    assert all(regs[a] is False for a in REGS)
