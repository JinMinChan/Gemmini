"""
Offline debug helper for "goal_success == 0.0 -> stop" cases.

This script does NOT touch the running server process. It reconstructs the RLRuntime
snapshot from a stored record JSON (app/records/<ip>/json/*.json), then re-runs MC
rollouts with larger N to see whether 0/128 was just sampling noise or truly ~0.

Usage:
  conda run -n gem python codex/debug_goal_success_zero.py \
    --record app/records/<ip>/json/<record>.json \
    --rollouts 128,512,2048,8192
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable


def _parse_int_list(csv: str) -> list[int]:
    out: list[int] = []
    for part in (csv or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _cp_upper_bound_zero_successes(n: int, alpha: float) -> float:
    # Clopper-Pearson upper bound when k=0.
    # P(X=0) = (1-p)^n >= alpha  =>  p <= 1 - alpha^(1/n)
    n = max(1, int(n))
    alpha = float(alpha)
    return 1.0 - math.exp(math.log(alpha) / float(n))


def _print_json(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", required=True, help="Path to record JSON")
    ap.add_argument(
        "--rollouts",
        default="128,512,2048,8192",
        help="Comma-separated rollout counts to test",
    )
    ap.add_argument("--seed", type=int, default=None, help="Optional fixed seed")
    args = ap.parse_args(argv)

    # When this script is run as `python codex/...py`, sys.path[0] becomes `codex/`,
    # so we need to explicitly add repo root for `import app.*`.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    record_path = Path(args.record)
    data = json.loads(record_path.read_text(encoding="utf-8"))

    manual_stats = dict(data.get("manual_stats") or {})
    ui_state = dict(data.get("ui_state") or {})
    ocr_result = dict(data.get("ocr_result") or {})
    goal = dict(((data.get("goal_success") or {}).get("target")) or {})

    if not goal:
        raise SystemExit("record has no goal_success.target")

    # Importing app.server is safe here: this script runs in its own process.
    from app.server import _get_rl_runtime  # noqa: WPS433 (runtime import by design)

    runtime = _get_rl_runtime()

    # Rebuild snapshot as server would do pre-override.
    rl = runtime.recommend(ocr_result=ocr_result, ui_state=ui_state, manual_stats=manual_stats)

    print("== Snapshot ==")
    print(f"record: {record_path}")
    print(f"goal: {goal}")
    print("state_used:")
    _print_json(rl.get("state_used") or {})
    print("options (ui_state first 4):")
    _print_json((ui_state.get("options") or [])[:4])
    print("rl recommend (pre any server override):")
    _print_json({k: rl.get(k) for k in ("action", "action_name", "confidence", "action_mask", "action_probs")})
    print()

    rollouts_list = _parse_int_list(args.rollouts)
    if not rollouts_list:
        rollouts_list = [128]

    print("== MC rollouts (policy-following) ==")
    for n in rollouts_list:
        r = runtime.estimate_goal_success_mc(goal, n_rollouts=int(n), seed=args.seed)
        sp = float(r.get("success_prob") or 0.0)
        ok = int(r.get("successes") or 0)
        nn = int(r.get("n_rollouts") or n)
        print(f"n={nn:>5}  successes={ok:>5}  p_hat={sp:.6f}")
        if ok == 0:
            ub95 = _cp_upper_bound_zero_successes(nn, 0.05)
            ub99 = _cp_upper_bound_zero_successes(nn, 0.01)
            print(f"         CP upper bound: 95%<= {ub95:.4%} , 99%<= {ub99:.4%}")
    print()

    # First-action probe at a moderate N so we can see whether "reroll first" helps at all.
    print("== MC probe (force first action, then follow policy) ==")
    r_probe = runtime.estimate_goal_success_mc_by_first_action(goal, n_rollouts=256, seed=args.seed)
    _print_json(r_probe)
    print()

    print("== MC strategy fallback (not policy) ==")
    for strat in ("always_process", "random_valid"):
        r_s = runtime.estimate_goal_success_mc_with_strategy(goal, n_rollouts=2048, strategy=strat, seed=args.seed)
        print(f"strategy={strat:>13}  successes={int(r_s.get('successes') or 0):>5}  p_hat={float(r_s.get('success_prob') or 0.0):.6f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
