"""
test_evidence_clips.py -- regression tests for forensic evidence clip capture.

Run:  python tests\test_evidence_clips.py     (from the repo root)

WHY THIS FILE EXISTS
--------------------
Evidence clips silently stopped finalizing: 26 of 29 recordings were stranded on
disk as "<name>.mp4.part.mp4" and their clip status stayed "recording" forever,
so the UI sat on "PROCESSING... Finalizing local MP4 evidence clip" and the
end-of-session report embedded a "still processing" placeholder instead of video.

Root cause: the ffmpeg encoder ran with stderr=subprocess.PIPE and nothing
reading it. ffmpeg filled the OS pipe buffer (~4 KB on Windows), blocked on
write, and stopped draining its stdin; the writer then blocked feeding frames.
proc.wait() never returned -> os.replace() never ran. Measured: wait() hung
indefinitely with 24 MB already encoded.

These tests lock in the observable contract, so the failure cannot come back
quietly:
  * a triggered clip reaches status "ready" within a bounded time,
  * the finalized file exists at the FINAL path with no ".part.mp4" left over,
  * the file is genuinely decodable H.264/yuv420p with faststart (i.e. a
    browser can actually play it -- an mp4v file would pass "file exists" but
    render as a black box),
  * ending a session finalizes in-flight clips instead of stranding them.

The encoder is exercised through the real capture_evidence_clip() path, not a
reimplementation, so a future edit to server.py is what gets tested.
"""

import os
import subprocess
import sys
import tempfile
import threading
import time

# Keep the import light and the test fast. These must be set BEFORE server is
# imported, because the module reads them at import time.
os.environ.setdefault("FACE_ID", "off")
os.environ.setdefault("PHONE_DETECTION", "off")
os.environ.setdefault("EVIDENCE_PRE_ROLL_SECONDS", "2")
os.environ.setdefault("EVIDENCE_POST_ROLL_SECONDS", "2")
os.environ.setdefault("EVIDENCE_COOLDOWN_SECONDS", "0")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "backend"))

import numpy as np  # noqa: E402

import server  # noqa: E402  (backend/server.py)


FRAME_H, FRAME_W = 480, 640


def _frame(seed):
    """A textured frame -- flat colour compresses to almost nothing and would
    hide size/duration regressions."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (FRAME_H, FRAME_W, 3), dtype=np.uint8)


def _fill_preroll(seconds=2.0, fps=15.0):
    """Populate the pre-roll ring buffer as the camera thread would."""
    now = time.time()
    n = int(seconds * fps)
    with server._evidence_buffer_lock:
        server._evidence_frame_buffer.clear()
        for i in range(n):
            ts = now - seconds + (i / fps)
            server._evidence_frame_buffer.append((ts, _frame(i)))
    return now


def _feed_live_frames(stop_event, fps=15.0):
    """Stand in for the camera worker during the post-roll window."""
    i = 1000
    while not stop_event.is_set():
        with server._raw_lock:
            server._latest_raw_frame = _frame(i)
            server._latest_raw_ts = time.time()
        i += 1
        time.sleep(1.0 / fps)


def _wait_for_status(clip_id, statuses, timeout=45.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with server._evidence_clips_lock:
            for c in server.session_evidence_clips:
                if c.get("id") == clip_id and c.get("status") in statuses:
                    return dict(c)
        time.sleep(0.05)
    with server._evidence_clips_lock:
        for c in server.session_evidence_clips:
            if c.get("id") == clip_id:
                return dict(c)
    return None


def _ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        import shutil
        return shutil.which("ffmpeg")


def _assert_browser_playable(path):
    """A file that exists is not the same as a file a browser will play."""
    exe = _ffmpeg()
    assert exe, "no ffmpeg available to verify the clip"
    probe = subprocess.run([exe, "-v", "error", "-i", path, "-f", "null", "-"],
                           capture_output=True, timeout=120)
    assert probe.returncode == 0, (
        "clip does not decode cleanly: "
        + probe.stderr.decode("utf-8", "replace")[:300])
    with open(path, "rb") as fh:
        head = fh.read(8192)
    assert b"avc1" in head, "clip is not H.264/avc1 (browsers won't play mp4v)"
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    assert moov != -1 and (mdat == -1 or moov < mdat), \
        "moov atom is not at the front (-movflags +faststart missing)"


def _begin_session(tmpdir):
    server.EVIDENCE_DIR = tmpdir
    server.SESSION_ACTIVE = True
    server._begin_evidence_session(True)


# ---------------------------------------------------------------------------
# Test 1: the core regression. A triggered clip must finalize to a real,
# playable .mp4 -- not hang on "recording" with a stranded .part.mp4.
# ---------------------------------------------------------------------------
def test_clip_finalizes_and_is_playable():
    tmpdir = tempfile.mkdtemp()
    _begin_session(tmpdir)
    trigger = _fill_preroll()

    stop = threading.Event()
    feeder = threading.Thread(target=_feed_live_frames, args=(stop,), daemon=True)
    feeder.start()
    try:
        clip = server.capture_evidence_clip("PHONE_DETECTED", trigger)
        assert clip is not None, "capture_evidence_clip returned None"
        final = _wait_for_status(clip["id"], {"ready", "failed"})
    finally:
        stop.set()

    assert final is not None, "clip vanished from the session index"
    assert final["status"] == "ready", (
        f"clip did not finalize: status={final['status']} "
        f"error={final.get('error')}")

    path = final["file_path"]
    assert os.path.exists(path), f"final mp4 missing at {path}"
    assert os.path.getsize(path) > 0, "final mp4 is empty"

    # The exact failure signature of the bug: temp file left behind.
    leftovers = []
    for root, _dirs, files in os.walk(tmpdir):
        leftovers += [f for f in files if f.endswith(".part.mp4")]
    assert not leftovers, f"stranded temp files left behind: {leftovers}"

    _assert_browser_playable(path)


# ---------------------------------------------------------------------------
# Test 2: encoding must not hang. The original bug's signature was an encoder
# that never returned, so assert a hard wall-clock bound on the whole capture.
# ---------------------------------------------------------------------------
def test_capture_completes_within_time_budget():
    tmpdir = tempfile.mkdtemp()
    _begin_session(tmpdir)
    trigger = _fill_preroll()

    stop = threading.Event()
    feeder = threading.Thread(target=_feed_live_frames, args=(stop,), daemon=True)
    feeder.start()
    t0 = time.time()
    try:
        clip = server.capture_evidence_clip("PHONE_DETECTED", trigger)
        final = _wait_for_status(clip["id"], {"ready", "failed"}, timeout=45)
    finally:
        stop.set()
    elapsed = time.time() - t0

    assert final and final["status"] == "ready", "clip did not finalize"
    # post-roll (2s) + encode. Generous ceiling; the bug made this unbounded.
    budget = server.POST_ROLL_SECONDS + 20.0
    assert elapsed < budget, (
        f"capture took {elapsed:.1f}s, over the {budget:.0f}s budget "
        f"(encoder may be hanging again)")


# ---------------------------------------------------------------------------
# Test 3: ending a session must finalize in-flight clips promptly, so the
# end-of-session report embeds video instead of a "still processing" note.
# ---------------------------------------------------------------------------
def test_session_end_finalizes_inflight_clip():
    tmpdir = tempfile.mkdtemp()
    _begin_session(tmpdir)
    trigger = _fill_preroll()

    stop = threading.Event()
    feeder = threading.Thread(target=_feed_live_frames, args=(stop,), daemon=True)
    feeder.start()
    try:
        clip = server.capture_evidence_clip("PHONE_DETECTED", trigger)
        assert clip is not None
        # End the session immediately -- mid post-roll, the exact race that
        # produced "Evidence capture is still processing" in the report.
        time.sleep(0.3)
        server.SESSION_ACTIVE = False
        server._end_evidence_session()

        t0 = time.time()
        drained = server._await_pending_evidence(timeout=40)
        waited = time.time() - t0
    finally:
        stop.set()

    assert drained, "clips were still recording after _await_pending_evidence"

    with server._evidence_clips_lock:
        states = [(c["id"], c["status"]) for c in server.session_evidence_clips]
    assert all(s == "ready" for _i, s in states), \
        f"in-flight clip not finalized at session end: {states}"
    # Post-roll should be cut short by session end, not run its full length.
    assert waited < server.POST_ROLL_SECONDS + 20.0, \
        f"session end waited {waited:.1f}s for evidence"

    for _cid, _s in states:
        pass
    with server._evidence_clips_lock:
        for c in server.session_evidence_clips:
            assert os.path.exists(c["file_path"]), \
                f"finalized clip missing on disk: {c['file_path']}"


# ---------------------------------------------------------------------------
# Test 4: only whitelisted device/material events may produce forensic video,
# and nothing is captured outside an active session.
# ---------------------------------------------------------------------------
def test_capture_is_scoped_to_devices_and_active_session():
    tmpdir = tempfile.mkdtemp()
    _begin_session(tmpdir)
    _fill_preroll()

    assert server.capture_evidence_clip("GAZE_DEVIATION") is None, \
        "a behavior alert must not create forensic video"
    assert server.capture_evidence_clip("FACE_MISSING") is None, \
        "an identity alert must not create forensic video"

    server.SESSION_ACTIVE = False
    assert server.capture_evidence_clip("PHONE_DETECTED") is None, \
        "no clip may be captured while no session is active"


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
            import traceback
            print(f"ERROR {t.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
