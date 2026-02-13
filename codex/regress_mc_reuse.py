"""
Regression test for MC rollout reuse bug (false 0% after many calls).

Before the fix: decision_steps wasn't reset on reused env instances, so after enough
calls every rollout truncated immediately and success_prob collapsed to 0.

This script runs estimate_goal_success_mc() repeatedly on the same snapshot and
fails if the estimate collapses to 0 after previously being > 0.

Usage:
  conda run -n gem python codex/regress_mc_reuse.py --record app/records/<ip>/json/<record>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--rollouts", type=int, default=128)
    args = ap.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    data = json.loads(Path(args.record).read_text(encoding="utf-8"))
    manual = dict(data.get("manual_stats") or {})
    ui = dict(data.get("ui_state") or {})
    ocr = dict(data.get("ocr_result") or {})
    goal = dict(((data.get("goal_success") or {}).get("target")) or {})
    if not goal:
        raise SystemExit("record has no goal_success.target")

    from app.server import _get_rl_runtime  # noqa: WPS433

    rt = _get_rl_runtime()
    rt.recommend(ocr_result=ocr, ui_state=ui, manual_stats=manual)

    n = max(1, int(args.rollouts))
    iters = max(1, int(args.iterations))

    ever_positive = False
    collapsed = False

    for i in range(iters):
        r = rt.estimate_goal_success_mc(goal, n_rollouts=n)
        sp = float(r.get("success_prob") or 0.0)
        ok = int(r.get("successes") or 0)
        seed = int(r.get("seed") or 0)
        print(f"iter={i+1:02d}  n={n:>4}  successes={ok:>4}  p_hat={sp:.6f}  seed={seed}")
        if sp > 0.0:
            ever_positive = True
        if ever_positive and sp <= 0.0:
            collapsed = True
            break

    if collapsed:
        print("FAIL: success_prob collapsed to 0 after being >0 (reuse/truncation regression).")
        return 1
    print("OK: no collapse detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

