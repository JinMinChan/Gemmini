from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces


def clamp_1_5(x: int) -> int:
    return max(1, min(5, int(x)))


@dataclass
class ProcessingOption:
    weight: float
    effect: Dict[str, int]
    exclude_condition: Callable[[Dict[str, int]], bool]
    tag: str


def weighted_sample_without_replacement(
    pool: Sequence[ProcessingOption], k: int, rng: np.random.Generator
) -> List[ProcessingOption]:
    items = list(pool)
    selected: List[ProcessingOption] = []
    for _ in range(min(int(k), len(items))):
        total = sum(float(o.weight) for o in items)
        if total <= 0:
            break
        r = float(rng.random()) * total
        acc = 0.0
        idx = 0
        for i, opt in enumerate(items):
            acc += float(opt.weight)
            if r <= acc:
                idx = i
                break
        selected.append(items.pop(idx))
    return selected


OPTION_TYPE_TO_ID = {
    "noop_keep": 0,
    "change_effect1": 1,
    "change_effect2": 2,
    "willpower": 3,
    "points": 4,
    "effect1_level": 5,
    "effect2_level": 6,
    "gold_state": 7,
    "rerolls": 8,
}
N_OPTION_TYPES = len(OPTION_TYPE_TO_ID)


def encode_option_onehot(opt: ProcessingOption) -> np.ndarray:
    vec = np.zeros(N_OPTION_TYPES + 1, dtype=np.float32)
    if not opt.effect:
        type_id = int(OPTION_TYPE_TO_ID.get(opt.tag, OPTION_TYPE_TO_ID["noop_keep"]))
        value = 0
    else:
        k, v = list(opt.effect.items())[0]
        type_id = int(OPTION_TYPE_TO_ID.get(k, OPTION_TYPE_TO_ID["noop_keep"]))
        value = int(v)
    vec[type_id] = 1.0
    vec[-1] = float(value)
    return vec


ROLE_TO_ID = {"dealer": 0, "support": 1}
ROLE_ID_TO_NAME = {v: k for k, v in ROLE_TO_ID.items()}

GEM_TYPE_TO_ID = {
    "stable": 0,  # 안정/침식
    "solid": 1,  # 견고/왜곡
    "immutable": 2,  # 불변/붕괴
}
GEM_TYPE_ID_TO_NAME = {v: k for k, v in GEM_TYPE_TO_ID.items()}

SUBOPT_TO_ID = {
    "attack": 0,  # 공격력
    "additional_damage": 1,  # 추가피해
    "boss_damage": 2,  # 보스피해
    "ally_damage_boost": 3,  # 아군피해강화
    "stigma": 4,  # 낙인력
    "ally_attack_boost": 5,  # 아군공격강화
}
SUBOPT_ID_TO_NAME = {v: k for k, v in SUBOPT_TO_ID.items()}

GEM_ALLOWED_SUBOPTS = {
    GEM_TYPE_TO_ID["stable"]: (
        SUBOPT_TO_ID["attack"],
        SUBOPT_TO_ID["additional_damage"],
        SUBOPT_TO_ID["ally_damage_boost"],
        SUBOPT_TO_ID["stigma"],
    ),
    GEM_TYPE_TO_ID["solid"]: (
        SUBOPT_TO_ID["attack"],
        SUBOPT_TO_ID["boss_damage"],
        SUBOPT_TO_ID["ally_damage_boost"],
        SUBOPT_TO_ID["ally_attack_boost"],
    ),
    GEM_TYPE_TO_ID["immutable"]: (
        SUBOPT_TO_ID["additional_damage"],
        SUBOPT_TO_ID["boss_damage"],
        SUBOPT_TO_ID["stigma"],
        SUBOPT_TO_ID["ally_attack_boost"],
    ),
}

ROLE_IMPORTANCE = {
    ROLE_TO_ID["dealer"]: {
        SUBOPT_TO_ID["attack"]: 1.0,
        SUBOPT_TO_ID["additional_damage"]: 2.0,
        SUBOPT_TO_ID["boss_damage"]: 3.0,
        SUBOPT_TO_ID["ally_damage_boost"]: 0.0,
        SUBOPT_TO_ID["stigma"]: 0.0,
        SUBOPT_TO_ID["ally_attack_boost"]: 0.0,
    },
    ROLE_TO_ID["support"]: {
        SUBOPT_TO_ID["attack"]: 0.0,
        SUBOPT_TO_ID["additional_damage"]: 0.0,
        SUBOPT_TO_ID["boss_damage"]: 0.0,
        SUBOPT_TO_ID["ally_damage_boost"]: 1.0,
        SUBOPT_TO_ID["stigma"]: 2.0,
        SUBOPT_TO_ID["ally_attack_boost"]: 3.0,
    },
}


class ArcGridGemRoleEnv(gym.Env):
    """
    Role/GemType-aware gem environment.

    Reward design (simplified):
    - Potential-based shaping with a simple scalar potential:
      Phi(s) = 2.5*willpower + 2.5*points + sub1_scale*effect1_level + sub2_scale*effect2_level
    - sub scale by role match:
      preferred suboption => 1.5 (base 1.0 + 0.5)
      non-preferred suboption => 0.5 (base 1.0 - 0.5)
    - Reroll/stop auxiliary terms exist for ablation, but defaults are 0.0.
    """

    ACTION_PROCESS = 0
    ACTION_REROLL = 1
    ACTION_STOP = 2

    def __init__(
        self,
        seed: Optional[int] = None,
        gamma: float = 0.99,
        shaping: bool = True,
        fixed_role: Optional[str] = None,
        fixed_gem_type: Optional[str] = None,
        util_w_total: float = 1.0,
        util_w_primary: float = 1.6,
        util_w_primary_balance: float = 1.0,
        util_w_role: float = 1.0,
        util_w_synergy: float = 0.8,
        action_w_reroll_adv: float = 0.0,
        action_w_stop_adv_penalty: float = 0.0,
        reward_primary: float = 50.0,
        reward_primary_goal_bonus: float = 0.0,
        reward_primary_overcap_scale: float = 1.0,
        reward_sub_total: float = 2.0,
        reward_role_bonus: float = 1.0,
        reward_reroll_quality: float = 1.0,
        reward_reroll_quality_endgame_multiplier: float = 1.0,
        reward_fail_unused_reroll_penalty: float = 0.0,
        reward_stop_counterfactual_coef: float = 1.0,
        stop_counterfactual_reroll_samples: int = 8,
        endgame_injection_prob: float = 0.0,
        endgame_attempts_left_min: int = 1,
        endgame_attempts_left_max: int = 2,
        endgame_reroll_bias: float = 0.7,
        endgame_reroll_trigger_attempts: int = 3,
        max_decision_steps: int = 64,
    ) -> None:
        super().__init__()

        self.gamma = float(gamma)
        self.shaping = bool(shaping)

        self.util_w_total = float(util_w_total)
        self.util_w_primary = float(util_w_primary)
        self.util_w_primary_balance = float(util_w_primary_balance)
        self.util_w_role = float(util_w_role)
        self.util_w_synergy = float(util_w_synergy)
        self.action_w_reroll_adv = float(max(0.0, action_w_reroll_adv))
        self.action_w_stop_adv_penalty = float(max(0.0, action_w_stop_adv_penalty))

        # Deprecated legacy reward fields are kept for backward compatibility only.
        if abs(float(reward_primary_goal_bonus)) > 1e-9 or abs(float(reward_primary_overcap_scale - 1.0)) > 1e-9:
            warnings.warn(
                "reward_primary_goal_bonus/reward_primary_overcap_scale are deprecated in v8 utility mode.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.reward_primary = float(reward_primary)
        self.reward_primary_goal_bonus = float(reward_primary_goal_bonus)
        self.reward_primary_overcap_scale = float(max(0.0, min(1.0, reward_primary_overcap_scale)))
        self.reward_sub_total = float(reward_sub_total)
        self.reward_role_bonus = float(reward_role_bonus)
        self.reward_reroll_quality = float(reward_reroll_quality)
        self.reward_reroll_quality_endgame_multiplier = max(0.0, float(reward_reroll_quality_endgame_multiplier))
        self.reward_fail_unused_reroll_penalty = max(0.0, float(reward_fail_unused_reroll_penalty))
        self.reward_stop_counterfactual_coef = max(0.0, float(reward_stop_counterfactual_coef))
        self.stop_counterfactual_reroll_samples = max(1, int(stop_counterfactual_reroll_samples))
        self.endgame_injection_prob = float(max(0.0, min(1.0, float(endgame_injection_prob))))
        self.endgame_attempts_left_min = max(1, int(endgame_attempts_left_min))
        self.endgame_attempts_left_max = max(1, int(endgame_attempts_left_max))
        if self.endgame_attempts_left_max < self.endgame_attempts_left_min:
            self.endgame_attempts_left_max = self.endgame_attempts_left_min
        self.endgame_reroll_bias = float(max(0.0, min(1.0, float(endgame_reroll_bias))))
        self.endgame_reroll_trigger_attempts = max(1, int(endgame_reroll_trigger_attempts))
        self.max_decision_steps = max(1, int(max_decision_steps))

        self.fixed_role_id = self._parse_role_id(fixed_role) if fixed_role is not None else None
        self.fixed_gem_type_id = self._parse_gem_type_id(fixed_gem_type) if fixed_gem_type is not None else None

        self.variant_settings = [
            {"max_attempts": 7, "initial_rerolls": 1},
            {"max_attempts": 9, "initial_rerolls": 2},
        ]

        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(3)

        # base:
        # [w,p,e1_lvl,e2_lvl,attemptsLeft,rerolls,gold_state] = 7
        # + role onehot(2) + gem_type onehot(3)
        # + effect1_kind onehot(6) + effect2_kind onehot(6)
        # => 24
        # option blocks: 4 * (onehot(9) + value(1)) = 40
        # total = 64
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(64,), dtype=np.float32)

        self.max_attempts = 9
        self.initial_rerolls = 2

        self.processing_options = self._build_processing_options()

        self.state: Dict[str, int] = {}
        self.current_options: List[ProcessingOption] = []
        self.terminated = False
        self.truncated = False
        self.first_process_done = False
        self.decision_steps = 0

    @staticmethod
    def _parse_role_id(role: str) -> int:
        s = str(role).strip().lower()
        if s not in ROLE_TO_ID:
            raise ValueError(f"unknown role: {role!r}; expected one of {sorted(ROLE_TO_ID.keys())}")
        return int(ROLE_TO_ID[s])

    @staticmethod
    def _parse_gem_type_id(gem_type: str) -> int:
        s = str(gem_type).strip().lower()
        if s not in GEM_TYPE_TO_ID:
            raise ValueError(f"unknown gem_type: {gem_type!r}; expected one of {sorted(GEM_TYPE_TO_ID.keys())}")
        return int(GEM_TYPE_TO_ID[s])

    def _sample_role_id(self) -> int:
        if self.fixed_role_id is not None:
            return int(self.fixed_role_id)
        return int(self.rng.integers(0, len(ROLE_TO_ID)))

    def _sample_gem_type_id(self) -> int:
        if self.fixed_gem_type_id is not None:
            return int(self.fixed_gem_type_id)
        return int(self.rng.integers(0, len(GEM_TYPE_TO_ID)))

    def _allowed_subopts(self, gem_type_id: int) -> Tuple[int, ...]:
        return tuple(GEM_ALLOWED_SUBOPTS[int(gem_type_id)])

    def _sample_distinct_kinds(self, allowed: Sequence[int]) -> Tuple[int, int]:
        allowed = list(allowed)
        if len(allowed) < 2:
            raise ValueError("allowed suboption set must contain at least 2 entries")
        picks = list(self.rng.choice(allowed, size=2, replace=False))
        return int(picks[0]), int(picks[1])

    def _has_change_candidates(self, s: Dict[str, int], slot: int) -> bool:
        gem_type_id = int(s["gem_type_id"])
        allowed = self._allowed_subopts(gem_type_id)
        if slot == 1:
            cur = int(s["effect1_kind"])
            other = int(s["effect2_kind"])
        else:
            cur = int(s["effect2_kind"])
            other = int(s["effect1_kind"])
        candidates = [k for k in allowed if k != cur and k != other]
        return len(candidates) > 0

    def _change_effect_kind(self, slot: int) -> None:
        s = self.state
        gem_type_id = int(s["gem_type_id"])
        allowed = self._allowed_subopts(gem_type_id)
        if slot == 1:
            cur_key, other_key = "effect1_kind", "effect2_kind"
        else:
            cur_key, other_key = "effect2_kind", "effect1_kind"
        cur = int(s[cur_key])
        other = int(s[other_key])
        candidates = [k for k in allowed if k != cur and k != other]
        if not candidates:
            return
        s[cur_key] = int(self.rng.choice(candidates))

    def _build_processing_options(self) -> List[ProcessingOption]:
        options: List[ProcessingOption] = []

        def add(weight: float, effect: Dict[str, int], cond: Callable[[Dict[str, int]], bool], tag: str) -> None:
            options.append(ProcessingOption(float(weight), effect, cond, tag))

        # willpower
        add(11.65, {"willpower": 1}, lambda s: s["willpower"] >= 5, "willpower")
        add(4.40, {"willpower": 2}, lambda s: s["willpower"] >= 4, "willpower")
        add(1.75, {"willpower": 3}, lambda s: s["willpower"] >= 3, "willpower")
        add(0.45, {"willpower": 4}, lambda s: s["willpower"] >= 2, "willpower")
        add(3.00, {"willpower": -1}, lambda s: s["willpower"] <= 1, "willpower")

        # points
        add(11.65, {"points": 1}, lambda s: s["points"] >= 5, "points")
        add(4.40, {"points": 2}, lambda s: s["points"] >= 4, "points")
        add(1.75, {"points": 3}, lambda s: s["points"] >= 3, "points")
        add(0.45, {"points": 4}, lambda s: s["points"] >= 2, "points")
        add(3.00, {"points": -1}, lambda s: s["points"] <= 1, "points")

        # effect1 level
        add(11.65, {"effect1_level": 1}, lambda s: s["effect1_level"] >= 5, "effect1_level")
        add(4.40, {"effect1_level": 2}, lambda s: s["effect1_level"] >= 4, "effect1_level")
        add(1.75, {"effect1_level": 3}, lambda s: s["effect1_level"] >= 3, "effect1_level")
        add(0.45, {"effect1_level": 4}, lambda s: s["effect1_level"] >= 2, "effect1_level")
        add(3.00, {"effect1_level": -1}, lambda s: s["effect1_level"] <= 1, "effect1_level")

        # effect2 level
        add(11.65, {"effect2_level": 1}, lambda s: s["effect2_level"] >= 5, "effect2_level")
        add(4.40, {"effect2_level": 2}, lambda s: s["effect2_level"] >= 4, "effect2_level")
        add(1.75, {"effect2_level": 3}, lambda s: s["effect2_level"] >= 3, "effect2_level")
        add(0.45, {"effect2_level": 4}, lambda s: s["effect2_level"] >= 2, "effect2_level")
        add(3.00, {"effect2_level": -1}, lambda s: s["effect2_level"] <= 1, "effect2_level")

        # effect kind changes (now actually functional)
        add(3.25, {}, lambda s: not self._has_change_candidates(s, 1), "change_effect1")
        add(3.25, {}, lambda s: not self._has_change_candidates(s, 2), "change_effect2")
        add(1.75, {}, lambda _s: False, "noop_keep")

        # gold state
        add(1.75, {"gold_state": +1}, lambda s: s["attemptsLeft"] <= 1 or s["gold_state"] >= 2, "gold_state")
        add(1.75, {"gold_state": -1}, lambda s: s["attemptsLeft"] <= 1 or s["gold_state"] <= 0, "gold_state")

        # rerolls
        add(2.50, {"rerolls": 1}, lambda s: s["attemptsLeft"] <= 1, "rerolls")
        add(0.75, {"rerolls": 2}, lambda s: s["attemptsLeft"] <= 1, "rerolls")
        return options

    def action_masks(self) -> List[bool]:
        if self.terminated or self.truncated:
            return [False, False, False]
        attempts = int(self.state["attemptsLeft"])
        rerolls = int(self.state["rerolls"])
        return [
            attempts > 0,
            rerolls > 0 and self.first_process_done,
            self.first_process_done,
        ]

    def _generate_options(self) -> List[ProcessingOption]:
        available = [o for o in self.processing_options if not o.exclude_condition(self.state)]
        opts = weighted_sample_without_replacement(available, 4, self.rng)

        def opt_type_id(o: ProcessingOption) -> int:
            if o.effect:
                k = next(iter(o.effect.keys()))
                return int(OPTION_TYPE_TO_ID.get(k, 999))
            return int(OPTION_TYPE_TO_ID.get(o.tag, 999))

        opts.sort(key=opt_type_id)
        return opts

    def _role_weight(self, role_id: int, subopt_id: int) -> float:
        return float(ROLE_IMPORTANCE.get(int(role_id), {}).get(int(subopt_id), 0.0))

    def _is_role_preferred_subopt(self, role_id: int, subopt_id: int) -> bool:
        return float(self._role_weight(int(role_id), int(subopt_id))) > 0.0

    def _role_subopt_scale(self, role_id: int, subopt_id: int) -> float:
        # Requested simple role scaling:
        # preferred suboption -> +0.5 on top of base 1.0 (=> 1.5),
        # non-preferred suboption -> -0.5 from base 1.0 (=> 0.5).
        return 1.5 if self._is_role_preferred_subopt(role_id, subopt_id) else 0.5

    def _role_cap(self, role_id: int, gem_type_id: int) -> float:
        _ = int(role_id)
        _ = int(gem_type_id)
        # Two suboption slots, each max level is 5.
        # Role-weighted term is now binary preferred/non-preferred.
        return 10.0

    @staticmethod
    def _primary_goal_hit(s: Dict[str, int]) -> bool:
        return int(s["willpower"]) >= 4 and int(s["points"]) >= 4

    def _state_total_norm(self, s: Dict[str, int]) -> float:
        total = (
            float(int(s["willpower"]))
            + float(int(s["points"]))
            + float(int(s["effect1_level"]))
            + float(int(s["effect2_level"]))
        )
        return float(max(0.0, min(1.0, total / 20.0)))

    def _state_primary_norm(self, s: Dict[str, int]) -> float:
        wp = float(min(int(s["willpower"]), 4))
        pt = float(min(int(s["points"]), 4))
        return float(max(0.0, min(1.0, (wp + pt) / 8.0)))

    def _state_primary_balance_norm(self, s: Dict[str, int]) -> float:
        wp = float(min(int(s["willpower"]), 4))
        pt = float(min(int(s["points"]), 4))
        return float(max(0.0, min(1.0, (wp * pt) / 16.0)))

    def _state_role_weighted(self, s: Dict[str, int]) -> float:
        role_id = int(s["role_id"])
        e1_kind = int(s["effect1_kind"])
        e2_kind = int(s["effect2_kind"])
        e1_lvl = float(int(s["effect1_level"]))
        e2_lvl = float(int(s["effect2_level"]))
        return float(
            (1.0 if self._is_role_preferred_subopt(role_id, e1_kind) else 0.0) * e1_lvl
            + (1.0 if self._is_role_preferred_subopt(role_id, e2_kind) else 0.0) * e2_lvl
        )

    def _state_role_norm(self, s: Dict[str, int]) -> float:
        role_weighted = self._state_role_weighted(s)
        cap = self._role_cap(int(s["role_id"]), int(s["gem_type_id"]))
        return float(max(0.0, min(1.0, role_weighted / cap)))

    def utility_components_from_state(self, s: Dict[str, int]) -> Dict[str, float]:
        wp = float(int(s["willpower"]))
        pt = float(int(s["points"]))
        e1_lvl = float(int(s["effect1_level"]))
        e2_lvl = float(int(s["effect2_level"]))
        role_id = int(s["role_id"])
        e1_kind = int(s["effect1_kind"])
        e2_kind = int(s["effect2_kind"])

        e1_scale = float(self._role_subopt_scale(role_id, e1_kind))
        e2_scale = float(self._role_subopt_scale(role_id, e2_kind))
        primary_term = float(2.5 * wp + 2.5 * pt)
        subopt_term = float((e1_scale * e1_lvl) + (e2_scale * e2_lvl))
        utility = float(primary_term + subopt_term)

        # Keep normalized diagnostics for logging dashboards.
        total_norm = self._state_total_norm(s)
        primary_norm = self._state_primary_norm(s)
        primary_balance_norm = self._state_primary_balance_norm(s)
        role_norm = self._state_role_norm(s)
        synergy_norm = float(primary_norm * role_norm)
        return {
            "utility_total": utility,
            "utility_primary_term": float(primary_term),
            "utility_subopt_term": float(subopt_term),
            "effect1_role_scale": float(e1_scale),
            "effect2_role_scale": float(e2_scale),
            "total_norm": float(total_norm),
            "primary_norm": float(primary_norm),
            "primary_balance_norm": float(primary_balance_norm),
            "role_norm": float(role_norm),
            "synergy_norm": float(synergy_norm),
            "role_weighted": float(self._state_role_weighted(s)),
            "role_cap": float(self._role_cap(int(s["role_id"]), int(s["gem_type_id"]))),
        }

    def utility_from_state(self, s: Dict[str, int]) -> float:
        return float(self.utility_components_from_state(s)["utility_total"])

    def _potential(self, s: Dict[str, int]) -> float:
        return self.utility_from_state(s)

    def _apply_effect_to_state(self, s: Dict[str, int], effect: Dict[str, int]) -> None:
        for k, v in effect.items():
            s[k] = int(s.get(k, 0)) + int(v)

    def _simulated_process_delta(self, s: Dict[str, int], opt: ProcessingOption) -> float:
        base = self._potential(s)
        s2 = dict(s)
        if opt.effect:
            self._apply_effect_to_state(s2, opt.effect)
        elif opt.tag == "change_effect1":
            allowed = self._allowed_subopts(int(s2["gem_type_id"]))
            cur = int(s2["effect1_kind"])
            other = int(s2["effect2_kind"])
            candidates = [k for k in allowed if k != cur and k != other]
            if candidates:
                s2["effect1_kind"] = int(self.rng.choice(candidates))
        elif opt.tag == "change_effect2":
            allowed = self._allowed_subopts(int(s2["gem_type_id"]))
            cur = int(s2["effect2_kind"])
            other = int(s2["effect1_kind"])
            candidates = [k for k in allowed if k != cur and k != other]
            if candidates:
                s2["effect2_kind"] = int(self.rng.choice(candidates))

        for k in ("willpower", "points", "effect1_level", "effect2_level"):
            s2[k] = clamp_1_5(int(s2.get(k, 1)))
        s2["gold_state"] = max(0, min(2, int(s2.get("gold_state", 1))))
        s2["rerolls"] = max(0, int(s2.get("rerolls", 0)))
        return float(self._potential(s2) - base)

    def _simulated_process_role_delta(self, s: Dict[str, int], opt: ProcessingOption) -> float:
        base_role = self._state_role_norm(s)
        s2 = dict(s)
        if opt.effect:
            self._apply_effect_to_state(s2, opt.effect)
        elif opt.tag == "change_effect1":
            allowed = self._allowed_subopts(int(s2["gem_type_id"]))
            cur = int(s2["effect1_kind"])
            other = int(s2["effect2_kind"])
            candidates = [k for k in allowed if k != cur and k != other]
            if candidates:
                s2["effect1_kind"] = int(self.rng.choice(candidates))
        elif opt.tag == "change_effect2":
            allowed = self._allowed_subopts(int(s2["gem_type_id"]))
            cur = int(s2["effect2_kind"])
            other = int(s2["effect1_kind"])
            candidates = [k for k in allowed if k != cur and k != other]
            if candidates:
                s2["effect2_kind"] = int(self.rng.choice(candidates))
        for k in ("effect1_level", "effect2_level"):
            s2[k] = clamp_1_5(int(s2.get(k, 1)))
        return float(self._state_role_norm(s2) - base_role)

    def _option_set_quality(self, s: Dict[str, int], opts: Sequence[ProcessingOption]) -> float:
        if not opts:
            return 0.0
        deltas = [self._simulated_process_delta(s, opt) for opt in opts]
        return float(np.mean(deltas))

    def _option_set_role_quality(self, s: Dict[str, int], opts: Sequence[ProcessingOption]) -> float:
        if not opts:
            return 0.0
        deltas = [self._simulated_process_role_delta(s, opt) for opt in opts]
        return float(np.mean(deltas))

    def _estimate_expected_option_quality(self, s: Dict[str, int], *, n_samples: int, role_only: bool = False) -> float:
        """
        Estimate E[quality(new_option_set)] under current state distribution.
        Uses temporary RNG/state snapshots and restores them after sampling.
        """
        n = max(1, int(n_samples))
        rng_state = copy.deepcopy(self.rng.bit_generator.state)
        state_backup = self.state
        opts_backup = self.current_options
        try:
            self.state = dict(s)
            vals: List[float] = []
            for _ in range(n):
                sampled_opts = self._generate_options()
                if role_only:
                    vals.append(float(self._option_set_role_quality(self.state, sampled_opts)))
                else:
                    vals.append(float(self._option_set_quality(self.state, sampled_opts)))
            return float(np.mean(vals)) if vals else 0.0
        finally:
            self.state = state_backup
            self.current_options = opts_backup
            self.rng.bit_generator.state = rng_state

    def estimate_action_advantages(self, s: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        if s is None:
            s = self.state
        q_process = float(self._option_set_quality(s, self.current_options))
        q_reroll = 0.0
        can_reroll = bool(int(s.get("rerolls", 0)) > 0 and self.first_process_done)
        if can_reroll:
            q_now = q_process
            q_after = self._estimate_expected_option_quality(
                s,
                n_samples=int(self.stop_counterfactual_reroll_samples),
            )
            q_reroll = float(max(0.0, q_after - q_now))
        q_stop = 0.0
        stop_gap = float(max(0.0, max(q_process, q_reroll) - q_stop))
        return {
            "q_process": float(q_process),
            "q_reroll": float(q_reroll),
            "q_stop": float(q_stop),
            "stop_advantage_gap": float(stop_gap),
        }

    def estimate_role_opportunity(self, s: Optional[Dict[str, int]] = None, n_samples: Optional[int] = None) -> Dict[str, float]:
        if s is None:
            s = self.state
        q_process_role = float(self._option_set_role_quality(s, self.current_options))
        q_reroll_role = 0.0
        can_reroll = bool(int(s.get("rerolls", 0)) > 0 and self.first_process_done)
        if can_reroll:
            q_after_role = self._estimate_expected_option_quality(
                s,
                n_samples=int(n_samples or self.stop_counterfactual_reroll_samples),
                role_only=True,
            )
            q_reroll_role = float(max(0.0, q_after_role - q_process_role))
        return {
            "q_process_role": float(q_process_role),
            "q_reroll_role": float(q_reroll_role),
            "high_role_upside": 1.0 if q_process_role > 0.0 else 0.0,
            "role_opportunity": 1.0 if q_reroll_role > 0.0 else 0.0,
        }

    def _stop_counterfactual_penalty(self, s: Dict[str, int]) -> tuple[float, float, float, float]:
        adv = self.estimate_action_advantages(s)
        penalty = float(
            self.action_w_stop_adv_penalty
            * self.reward_stop_counterfactual_coef
            * float(adv["stop_advantage_gap"])
        )
        return penalty, float(adv["q_process"]), float(adv["q_reroll"]), float(adv["stop_advantage_gap"])

    def _inject_endgame_start_if_enabled(self) -> None:
        """
        Optional curriculum helper: after a fresh reset, run a short random
        prefix to generate late-game states more frequently.
        """
        if self.endgame_injection_prob <= 0.0:
            return
        if float(self.rng.random()) >= self.endgame_injection_prob:
            return

        target_min = max(1, int(self.endgame_attempts_left_min))
        target_max = max(target_min, int(self.endgame_attempts_left_max))
        target_max = min(target_max, max(1, int(self.max_attempts) - 1))
        target_attempts_left = int(self.rng.integers(target_min, target_max + 1))

        guard = 0
        while int(self.state.get("attemptsLeft", 0)) > target_attempts_left and guard < 128:
            guard += 1
            masks = self.action_masks()
            if not masks[self.ACTION_PROCESS]:
                break

            attempts_now = int(self.state.get("attemptsLeft", 0))
            choose_reroll = bool(
                masks[self.ACTION_REROLL]
                and attempts_now <= int(self.endgame_reroll_trigger_attempts)
                and float(self.rng.random()) < float(self.endgame_reroll_bias)
            )
            action = self.ACTION_REROLL if choose_reroll else self.ACTION_PROCESS
            _obs, _reward, term, trunc, _info = self.step(int(action))
            if term or trunc:
                break

        # Burn-in is only for start-state generation; clear control flags.
        if self.terminated or self.truncated:
            self.terminated = False
            self.truncated = False
        self.decision_steps = 0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        variant = self.rng.choice(self.variant_settings)
        self.max_attempts = int(variant["max_attempts"])
        self.initial_rerolls = int(variant["initial_rerolls"])

        role_id = self._sample_role_id()
        gem_type_id = self._sample_gem_type_id()
        allowed = self._allowed_subopts(gem_type_id)
        e1_kind, e2_kind = self._sample_distinct_kinds(allowed)

        self.state = {
            "willpower": 1,
            "points": 1,
            "effect1_level": 1,
            "effect2_level": 1,
            "attemptsLeft": self.max_attempts,
            "rerolls": self.initial_rerolls,
            "gold_state": 1,
            "role_id": int(role_id),
            "gem_type_id": int(gem_type_id),
            "effect1_kind": int(e1_kind),
            "effect2_kind": int(e2_kind),
        }

        self.first_process_done = False
        self.terminated = False
        self.truncated = False
        self.decision_steps = 0
        self.current_options = self._generate_options()

        # Stage-aware curriculum: expose more endgame states during training.
        self._inject_endgame_start_if_enabled()
        return self._get_obs(), {}

    def step(self, action: int):
        phi_before = self._potential(self.state)
        reward = 0.0
        # NOTE: This info dict is used by training and also by the HumanData UI.
        # Keep keys stable; values may include ints/bools for step metadata.
        info: Dict[str, Any] = {}
        self.decision_steps += 1
        reroll_advantage = 0.0
        stop_adv_gap = 0.0
        q_process = 0.0
        q_reroll = 0.0
        attempts_before = int(self.state.get("attemptsLeft", 0))

        if int(action) == self.ACTION_PROCESS:
            self.first_process_done = True
            chosen = self.rng.choice(self.current_options)
            try:
                chosen_idx = int(list(self.current_options).index(chosen))
            except ValueError:
                chosen_idx = -1
            info["step/chosen_option_idx"] = int(chosen_idx)
            info["step/chosen_option_tag"] = str(getattr(chosen, "tag", ""))
            if getattr(chosen, "effect", None):
                try:
                    k, v = list(chosen.effect.items())[0]
                    info["step/chosen_effect_key"] = str(k)
                    info["step/chosen_effect_value"] = int(v)
                except Exception:
                    pass

            if chosen.effect:
                self._apply_effect_to_state(self.state, chosen.effect)
            elif chosen.tag == "change_effect1":
                self._change_effect_kind(slot=1)
            elif chosen.tag == "change_effect2":
                self._change_effect_kind(slot=2)

            for k in ("willpower", "points", "effect1_level", "effect2_level"):
                self.state[k] = clamp_1_5(self.state[k])

            self.state["gold_state"] = max(0, min(2, int(self.state["gold_state"])))
            self.state["rerolls"] = max(0, int(self.state["rerolls"]))

            self.state["attemptsLeft"] = int(self.state["attemptsLeft"]) - 1
            if int(self.state["attemptsLeft"]) <= 0:
                self.terminated = True
            else:
                self.current_options = self._generate_options()

        elif int(action) == self.ACTION_REROLL:
            q_before = self._option_set_quality(self.state, self.current_options)
            self.state["rerolls"] = max(0, int(self.state["rerolls"]) - 1)
            self.current_options = self._generate_options()
            q_after = self._option_set_quality(self.state, self.current_options)
            reroll_advantage = float(max(0.0, q_after - q_before))

        elif int(action) == self.ACTION_STOP:
            self.terminated = True

        phi_after = self._potential(self.state)

        if self.shaping:
            if not self.terminated:
                reward = (self.gamma * phi_after) - phi_before
            else:
                reward = phi_after
        else:
            reward = phi_after if self.terminated else 0.0

        if self.shaping and int(action) == self.ACTION_REROLL:
            reward += (
                self.action_w_reroll_adv
                * self.reward_reroll_quality
                * float(self.reward_reroll_quality_endgame_multiplier)
                * float(reroll_advantage)
            )

        # Extra learning signal:
        # - discourage stop only when continuing has a better expected immediate value.
        # - optional penalty for ending with unused rerolls on failed primary progression.
        if self.shaping:
            if int(action) == self.ACTION_STOP and attempts_before > 0 and self.reward_stop_counterfactual_coef > 0.0:
                penalty, q_process, q_reroll, stop_adv_gap = self._stop_counterfactual_penalty(self.state)
                reward -= float(penalty)
                info["rewards/stop_counterfactual_penalty"] = float(penalty)
                info["rewards/stop_counterfactual_q_process"] = float(q_process)
                info["rewards/stop_counterfactual_q_reroll"] = float(q_reroll)
                info["rewards/stop_advantage_gap"] = float(stop_adv_gap)

            primary_ok = self._primary_goal_hit(self.state)

            if (
                self.terminated
                and int(self.state.get("attemptsLeft", 0)) <= 0
                and not primary_ok
                and int(self.state.get("rerolls", 0)) > 0
            ):
                penalty = self.reward_fail_unused_reroll_penalty * float(int(self.state.get("rerolls", 0)))
                reward -= float(penalty)

        info["rewards/reroll_advantage"] = float(reroll_advantage)

        if (not self.terminated) and (not self.truncated) and self.decision_steps >= self.max_decision_steps:
            self.truncated = True

        if self.terminated:
            role_id = int(self.state["role_id"])
            gem_type_id = int(self.state["gem_type_id"])
            e1_kind = int(self.state["effect1_kind"])
            e2_kind = int(self.state["effect2_kind"])
            comps = self.utility_components_from_state(self.state)
            term_info: Dict[str, Any] = {
                "rewards/phi_final": float(phi_after),
                "rewards/utility_total": float(comps["utility_total"]),
                "rewards/utility_total_norm": float(comps["total_norm"]),
                "rewards/utility_primary": float(comps["utility_primary_term"]),
                "rewards/utility_primary_balance": float(comps["primary_balance_norm"]),
                "rewards/utility_role": float(comps["utility_subopt_term"]),
                "rewards/utility_synergy": float(comps["synergy_norm"]),
                "rewards/effect1_role_scale": float(comps["effect1_role_scale"]),
                "rewards/effect2_role_scale": float(comps["effect2_role_scale"]),
                "rewards/role_weighted": float(comps["role_weighted"]),
                "rewards/role_cap": float(comps["role_cap"]),
                # Backward-compatible keys
                "rewards/primary_progress_component": float(comps["primary_norm"]),
                "rewards/primary_goal_bonus": 0.0,
                "rewards/sub_total_component": float(comps["total_norm"]),
                "rewards/role_component": float(comps["role_norm"]),
                "rewards/primary_goal_hit": float(1.0 if self._primary_goal_hit(self.state) else 0.0),
                "rewards/stop_counterfactual_coef": float(self.reward_stop_counterfactual_coef),
                "rewards/stop_advantage_gap": float(stop_adv_gap),
                "state/willpower": int(self.state["willpower"]),
                "state/points": int(self.state["points"]),
                "state/effect1_level": int(self.state["effect1_level"]),
                "state/effect2_level": int(self.state["effect2_level"]),
                "state/gold_state": int(self.state["gold_state"]),
                "state/role_id": int(role_id),
                "state/gem_type_id": int(gem_type_id),
                "state/effect1_kind_id": int(e1_kind),
                "state/effect2_kind_id": int(e2_kind),
                "state/process_count": int(self.max_attempts - int(self.state["attemptsLeft"])),
                "state/rerolls_used": int(self.initial_rerolls - int(self.state["rerolls"])),
            }
            info.update(term_info)
        elif self.truncated:
            trunc_info: Dict[str, Any] = {
                "state/willpower": int(self.state["willpower"]),
                "state/points": int(self.state["points"]),
                "state/effect1_level": int(self.state["effect1_level"]),
                "state/effect2_level": int(self.state["effect2_level"]),
                "state/gold_state": int(self.state["gold_state"]),
                "state/role_id": int(self.state["role_id"]),
                "state/gem_type_id": int(self.state["gem_type_id"]),
                "state/effect1_kind_id": int(self.state["effect1_kind"]),
                "state/effect2_kind_id": int(self.state["effect2_kind"]),
                "state/process_count": int(self.max_attempts - int(self.state["attemptsLeft"])),
                "state/rerolls_used": int(self.initial_rerolls - int(self.state["rerolls"])),
                "state/truncated_step_cap": int(self.max_decision_steps),
            }
            info.update(trunc_info)
        return self._get_obs(), float(reward), self.terminated, self.truncated, info

    def reset_from_state(self, state: Dict[str, int]):
        self.reset()
        self.state = {k: int(v) for k, v in state.items()}
        self.state["willpower"] = clamp_1_5(self.state.get("willpower", 1))
        self.state["points"] = clamp_1_5(self.state.get("points", 1))
        self.state["effect1_level"] = clamp_1_5(self.state.get("effect1_level", 1))
        self.state["effect2_level"] = clamp_1_5(self.state.get("effect2_level", 1))
        self.state["attemptsLeft"] = max(0, int(self.state.get("attemptsLeft", self.max_attempts)))
        self.state["rerolls"] = max(0, int(self.state.get("rerolls", self.initial_rerolls)))
        self.state["gold_state"] = max(0, min(2, int(self.state.get("gold_state", 1))))
        self.state["role_id"] = int(self.state.get("role_id", self._sample_role_id()))
        self.state["gem_type_id"] = int(self.state.get("gem_type_id", self._sample_gem_type_id()))
        allowed = self._allowed_subopts(self.state["gem_type_id"])
        e1_kind = int(self.state.get("effect1_kind", allowed[0]))
        e2_kind = int(self.state.get("effect2_kind", allowed[1]))
        if e1_kind == e2_kind:
            candidates = [k for k in allowed if k != e1_kind]
            if candidates:
                e2_kind = int(self.rng.choice(candidates))
        self.state["effect1_kind"] = int(e1_kind)
        self.state["effect2_kind"] = int(e2_kind)

        self.terminated = False
        self.truncated = False
        self.decision_steps = 0
        self.first_process_done = int(self.state["attemptsLeft"]) < int(self.max_attempts)
        self.current_options = self._generate_options()
        return self._get_obs()

    @staticmethod
    def _one_hot(idx: int, size: int) -> np.ndarray:
        v = np.zeros(size, dtype=np.float32)
        if 0 <= int(idx) < size:
            v[int(idx)] = 1.0
        return v

    def _get_obs(self) -> np.ndarray:
        s = self.state
        base = np.concatenate(
            [
                np.array(
                    [
                        s["willpower"],
                        s["points"],
                        s["effect1_level"],
                        s["effect2_level"],
                        s["attemptsLeft"],
                        s["rerolls"],
                        s["gold_state"],
                    ],
                    dtype=np.float32,
                ),
                self._one_hot(int(s["role_id"]), len(ROLE_TO_ID)),
                self._one_hot(int(s["gem_type_id"]), len(GEM_TYPE_TO_ID)),
                self._one_hot(int(s["effect1_kind"]), len(SUBOPT_TO_ID)),
                self._one_hot(int(s["effect2_kind"]), len(SUBOPT_TO_ID)),
            ],
            axis=0,
        )
        opt_vecs = np.concatenate([encode_option_onehot(o) for o in self.current_options], axis=0)
        obs = np.concatenate([base, opt_vecs], axis=0)
        return obs.astype(np.float32)
