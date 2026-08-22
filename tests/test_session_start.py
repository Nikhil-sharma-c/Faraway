"""
test_session_start.py -- regression test for POST /api/session/start.

Run:  python tests\test_session_start.py     (from the repo root)

WHY THIS FILE EXISTS
--------------------
Clicking "Start Examination Monitoring" returned 500 and the UI showed
"Failed to initialize session". Root cause: start_session reads the module
global `current_students_in_frame`, but that name was ONLY ever created by an
assignment inside _ai_worker_loop. On a freshly launched server -- Start clicked
from the enrollment page before the monitoring video feed has driven the AI loop
through its first assignment -- the name did not exist yet, so line
`total_faces = len(current_students_in_frame) + ...` raised
NameError -> HTTP 500.

This test imports the server WITHOUT ever running the camera or the AI loop
(threads only start under __main__), so `current_students_in_frame` is only
defined if it is initialised at module scope. It asserts the endpoint returns
200 in that cold state -- exactly the condition that used to 500.
"""

import os
import sys

os.environ.setdefault("FACE_ID", "off")
os.environ.setdefault("PHONE_DETECTION", "off")

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "backend"))
os.chdir(os.path.join(_ROOT, "backend"))

import server  # noqa: E402


def test_module_global_exists_before_ai_loop_runs():
    # The AI loop has never run in this process; the name must still exist.
    assert hasattr(server, "current_students_in_frame"), \
        "current_students_in_frame must be defined at module scope"
    assert isinstance(server.current_students_in_frame, set)


def test_session_start_returns_200_on_cold_server():
    client = server.app.test_client()
    resp = client.post("/api/session/start")
    assert resp.status_code == 200, (
        f"expected 200, got {resp.status_code}: "
        f"{resp.get_data(as_text=True)[:300]}")
    body = resp.get_json() or {}
    assert body.get("success") is True, f"unexpected body: {body}"
    # leave the process in a clean state for any following test
    client.post("/api/session/end")


def test_session_start_still_guards_multi_person():
    # The one-person rule must survive the fix: with two faces already counted,
    # start is rejected with 400 rather than silently starting.
    server.SESSION_ACTIVE = False
    server.accumulated_elapsed_seconds = 0
    server.current_students_in_frame = {"11011", "22022"}
    try:
        resp = server.app.test_client().post("/api/session/start")
        assert resp.status_code == 400, \
            f"multi-person start should be 400, got {resp.status_code}"
    finally:
        server.current_students_in_frame = set()


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
