"""
test_id_stabilizer.py -- permanent regression tests for identity/track stability.

Pure-python: no camera, no vision models. Runs with plain `python
test_id_stabilizer.py` or under pytest (`pytest test_id_stabilizer.py`).

WHY THIS FILE EXISTS
--------------------
Tracking continuity has now regressed TWICE from changes elsewhere in the
system (first the real-time tracking work, then the phone-detection /
pipeline-stability work). Each time the symptom was the same: an
already-confirmed, continuously-present person whose on-video tag cycled
"confirmed -> IDENTIFYING... -> confirmed" (or vanished entirely) because the
committed identity was being dropped even though the underlying track never
changed.

These tests lock in the contract that prevents that regression class:

  * Once a track is CONFIRMED to an enrolled identity, it stays committed
    ("known") for the entire life of the track -- across arbitrarily long runs
    of missed / dipped recognitions -- as long as the person stays in frame.
  * The track id stays constant the whole time.
  * The "IDENTIFYING..." (pending) state still appears exactly once for a
    genuinely NEW track, and a genuinely DIFFERENT identity still switches --
    the fix holds a confirmed identity, it does not suppress the pending state
    or freeze identity forever.

The stabiliser is deliberately decoupled from the vision models so this whole
contract is testable deterministically at simulated frame rates.
"""

import os
import sys

# Importable whether this file sits in tests/ (repo root on path -> ml package)
# or directly beside id_stabilizer.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))   # repo root, for `from ml import ...`
sys.path.insert(0, _HERE)
try:
    from ml import id_stabilizer  # noqa: E402
except ImportError:               # pragma: no cover
    import id_stabilizer          # noqa: E402


FPS = 15.0
DT = 1.0 / FPS


class Clock:
    """A single monotonic SIMULATED clock. Every update() in a test must be
    driven from the same Clock -- never `now=None` (wall clock), which does not
    advance in a tight loop and would make dwell/staleness windows meaningless.
    """
    def __init__(self):
        self.t = 0.0

    def tick(self, dt=DT):
        self.t += dt
        return self.t


def _step(stab, clk, faces, n, dt=DT):
    """Advance `n` frames of `faces` on clock `clk`; return per-frame outputs."""
    return [stab.update(faces, now=clk.tick(dt)) for _ in range(n)]


def _single_face(sid=None, name=None, cx=320.0, cy=240.0, size=120.0):
    return {"cx": cx, "cy": cy, "size": size, "sid": sid, "name": name}


# ---------------------------------------------------------------------------
# Test 1 + 2 (acceptance cases 1 & 2): sustained single-person stability.
# A known person is confirmed once, then stays continuously in frame for
# several simulated minutes while recognition only lands intermittently (long
# gaps, and one very long dropout where the identifier never sees the face).
# The track id must stay constant and the overlay state must stay 'known' with
# the same committed identity for the ENTIRE duration -- zero reverts to
# pending/unknown, zero moments of no identity.
# ---------------------------------------------------------------------------
def test_sustained_single_person_never_reverts():
    stab = id_stabilizer.IdentityStabilizer()
    minutes = 5
    total = int(minutes * 60 * FPS)
    confirm_every = 2.5          # a confident match only every 2.5 s
    # A long recognition blackout in the middle: identifier returns None for a
    # continuous 30 s stretch (person stays put but is never re-recognised).
    blackout = (int(120 * FPS), int(150 * FPS))

    t = 0.0
    last_confirm = -1e9
    import math
    states, tids, sids = [], [], []
    for i in range(total):
        t += DT
        in_blackout = blackout[0] <= i < blackout[1]
        if (not in_blackout) and (t - last_confirm) >= confirm_every:
            raw_sid, raw_name = "11011", "Royce Dcunha"
            last_confirm = t
        else:
            raw_sid, raw_name = None, None
        # Small natural head jitter so association is exercised, not a fixed pt.
        cx = 320.0 + 12.0 * math.sin(i / 9.0)
        cy = 240.0 + 8.0 * math.cos(i / 11.0)
        out = stab.update([_single_face(raw_sid, raw_name, cx, cy)], now=t)[0]
        states.append(out["state"])
        tids.append(out["tid"])
        sids.append(out["sid"])

    first_known = next((i for i, s in enumerate(states) if s == "known"), None)
    assert first_known is not None, "person was never confirmed at all"
    # Became known quickly (first confirm lands ~frame 0-1 here).
    assert first_known <= 1, f"confirmation took too long: frame {first_known}"

    # Track id constant throughout (acceptance case 2).
    assert len(set(tids)) == 1, f"track id churned: {sorted(set(tids))}"

    # Zero reverts out of 'known' after first becoming known (acceptance 1).
    tail_states = states[first_known:]
    assert all(s == "known" for s in tail_states), (
        "overlay reverted out of 'known' while person stayed in frame: "
        + " ".join(sorted(set(tail_states)))
    )
    # Committed identity persisted against that stable track the whole time,
    # including through the 30 s recognition blackout.
    tail_sids = sids[first_known:]
    assert all(s == "11011" for s in tail_sids), "committed identity was lost"


# ---------------------------------------------------------------------------
# Test 3a (acceptance case 3): a genuinely NEW person, after the previous
# person truly left frame (a real gap > TRACK_STALE_S), gets a fresh track and
# goes through the "IDENTIFYING..." (pending) state exactly once, then
# stabilises. Confirms the pending state still works for legitimate new tracks.
# ---------------------------------------------------------------------------
def test_new_person_after_real_gap_reidentifies_once():
    stab = id_stabilizer.IdentityStabilizer()
    clk = Clock()

    # Person A present and confirmed for a while.
    outs_a = _step(stab, clk, [_single_face("11011", "Royce Dcunha")],
                   int(5 * FPS))
    tid_a = outs_a[-1][0]["tid"]
    assert outs_a[-1][0]["state"] == "known"

    # Nobody in frame for longer than TRACK_STALE_S -> A's track is pruned.
    gap_frames = int((id_stabilizer.TRACK_STALE_S + 1.0) * FPS)
    _step(stab, clk, [], gap_frames)

    # Person B enters (unidentified at first), then gets recognised.
    b_states, b_tids = [], []
    for i in range(int(6 * FPS)):
        # B is only recognised after ~1 s of being present.
        recognised = i >= int(1.0 * FPS)
        f = _single_face("22022" if recognised else None,
                         "Alex Rivera" if recognised else None)
        out = stab.update([f], now=clk.tick())[0]
        b_states.append(out["state"])
        b_tids.append(out["tid"])

    tid_b = b_tids[-1]
    assert tid_b != tid_a, "new person reused the departed person's track id"
    assert "pending" in b_states, "new track never showed the IDENTIFYING state"
    assert b_states[0] == "pending", "brand-new track should start pending"
    # Ends confirmed to B and stays there.
    assert b_states[-1] == "known"
    # Exactly one pending->known transition (identified once, then stable).
    transitions = sum(1 for i in range(1, len(b_states))
                      if b_states[i] == "known" and b_states[i - 1] != "known")
    assert transitions == 1, f"expected one confirm, saw {transitions}"


# ---------------------------------------------------------------------------
# Test 3b: the SAME person re-entering after a real gap is treated as a new
# track and re-confirmed once -- not silently resurrected, not left unknown.
# ---------------------------------------------------------------------------
def test_same_person_reentry_after_gap_is_new_track():
    stab = id_stabilizer.IdentityStabilizer()
    clk = Clock()
    outs1 = _step(stab, clk, [_single_face("11011", "Royce")], int(4 * FPS))
    tid1 = outs1[-1][0]["tid"]

    _step(stab, clk, [], int((id_stabilizer.TRACK_STALE_S + 1.0) * FPS))

    # Re-enters; unidentified for the first 0.5 s, then recognised again.
    states, tids = [], []
    for i in range(int(4 * FPS)):
        recognised = i >= int(0.5 * FPS)
        f = _single_face("11011" if recognised else None,
                         "Royce" if recognised else None)
        out = stab.update([f], now=clk.tick())[0]
        states.append(out["state"]); tids.append(out["tid"])

    assert tids[-1] != tid1, "re-entry should get a fresh track id"
    assert states[0] == "pending" and states[-1] == "known"


# ---------------------------------------------------------------------------
# Test 4: a genuinely DIFFERENT confident identity on the SAME continuous track
# (person swap with no gap) must switch identity promptly -- the fix holds a
# confirmed identity through MISSES, it does not freeze identity against a real,
# confident re-identification.
# ---------------------------------------------------------------------------
def test_genuine_identity_change_switches():
    stab = id_stabilizer.IdentityStabilizer()
    clk = Clock()
    # A confirmed.
    out = _step(stab, clk, [_single_face("11011", "Royce")], int(3 * FPS))[-1][0]
    assert out["sid"] == "11011"
    # Same spatial track keeps getting a DIFFERENT confident id.
    out = _step(stab, clk, [_single_face("22022", "Alex")], int(2 * FPS))[-1][0]
    assert out["state"] == "known"
    assert out["sid"] == "22022", "genuine identity change was suppressed"


# ---------------------------------------------------------------------------
# Test 5: a NEVER-identified track escalates pending -> unknown after the dwell
# window (the identifying state is not suppressed, and genuine strangers still
# surface). Guards against "fixing" the flicker by just forcing everything known.
# ---------------------------------------------------------------------------
def test_unidentified_track_becomes_unknown():
    stab = id_stabilizer.IdentityStabilizer()
    clk = Clock()
    n = int((id_stabilizer.NEW_UNKNOWN_DWELL_S + 1.0) * FPS)
    states = [o[0]["state"] for o in
              _step(stab, clk, [_single_face(None, None)], n)]
    assert states[0] == "pending"
    assert states[-1] == "unknown", "sustained stranger never escalated to unknown"
    assert "known" not in states, "an unrecognised face was wrongly marked known"


# ---------------------------------------------------------------------------
# Test 6: brief sub-TRACK_STALE_S observation gaps (a couple of missed
# MediaPipe frames on a fast head turn) must NOT create a new track or drop the
# identity -- the same person re-associates to the same track and stays known.
# ---------------------------------------------------------------------------
def test_brief_observation_gap_keeps_track_and_identity():
    stab = id_stabilizer.IdentityStabilizer()
    clk = Clock()
    out = _step(stab, clk, [_single_face("11011", "Royce")], int(3 * FPS))[-1][0]
    tid = out["tid"]

    # ~0.5 s with no face observed at all (< TRACK_STALE_S).
    _step(stab, clk, [], int(0.5 * FPS))

    # Face reappears near the same place, unidentified this frame.
    out = stab.update([_single_face(None, None)], now=clk.tick())[0]
    assert out["tid"] == tid, "brief gap spawned a new track"
    assert out["state"] == "known", "brief gap dropped the confirmed identity"
    assert out["sid"] == "11011"


def _main():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception as e:
            failed += 1
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
