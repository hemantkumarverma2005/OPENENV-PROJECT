"""
curriculum.py — Auto-Curriculum & Difficulty Scaling
════════════════════════════════════════════════════
Implements adaptive difficulty scaling for SocialContract-v0.
When the agent scores above a threshold, the environment automatically
increases difficulty — creating a self-play/self-improvement loop.

This strengthens alignment with Theme #4 (Self-Improvement) and
demonstrates recursive skill amplification.

Difficulty axes:
  1. Shock frequency (more frequent economic shocks)
  2. Shock severity (larger GDP/inflation/unrest impacts)
  3. Grading strictness (tighter thresholds for gates)
  4. Citizen sensitivity (citizens react more strongly to policy changes)
  5. Initial conditions (start in worse economic states)
  6. Time pressure (fewer steps to achieve goals)

Academic inspiration:
  - Bengio et al. (2009) "Curriculum Learning"
  - OpenAI (2019) "Solving Rubik's Cube with a Robot Hand" — ADR
  - Sukhbaatar et al. (2018) "Intrinsic Motivation and Automatic Curricula"
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class DifficultyConfig:
    """Configuration for a specific difficulty level."""
    level: int = 0
    name: str = "Normal"

    # Shock parameters
    shock_frequency_mult: float = 1.0     # Multiplier on shock probability
    shock_severity_mult: float = 1.0      # Multiplier on shock effects
    shock_duration_bonus: int = 0          # Extra steps added to shock duration

    # Grading strictness
    grading_strictness: float = 1.0       # 1.0 = normal, >1 = stricter gates

    # Citizen sensitivity
    citizen_trust_decay_mult: float = 1.0  # How fast trust erodes
    citizen_unrest_sensitivity: float = 1.0  # How easily citizens get upset
    capital_flight_sensitivity: float = 1.0  # How easily ultra-rich flee

    # Initial condition modifiers
    initial_budget_penalty: float = 0.0    # Additional deficit at start
    initial_inflation_add: float = 0.0     # Additional inflation at start
    initial_unemployment_add: float = 0.0  # Additional unemployment at start

    # Time pressure
    max_steps_reduction: int = 0           # Fewer steps to complete task

    # Market agent
    market_aggressiveness: float = 0.3     # How aggressive market speculator is

    def describe(self) -> str:
        """Human-readable description of this difficulty level."""
        mods = []
        if self.shock_frequency_mult > 1.0:
            mods.append(f"shocks {self.shock_frequency_mult:.1f}x more frequent")
        if self.shock_severity_mult > 1.0:
            mods.append(f"shocks {self.shock_severity_mult:.1f}x more severe")
        if self.grading_strictness > 1.0:
            mods.append(f"grading {self.grading_strictness:.1f}x stricter")
        if self.citizen_trust_decay_mult > 1.0:
            mods.append(f"trust decays {self.citizen_trust_decay_mult:.1f}x faster")
        if self.max_steps_reduction > 0:
            mods.append(f"{self.max_steps_reduction} fewer steps")
        if self.market_aggressiveness > 0.5:
            mods.append(f"aggressive markets ({self.market_aggressiveness:.1f})")
        if not mods:
            return f"Level {self.level}: {self.name} (standard difficulty)"
        return f"Level {self.level}: {self.name} ({', '.join(mods)})"


# ── Predefined difficulty levels ─────────────────────────────────────────────

DIFFICULTY_LEVELS = [
    DifficultyConfig(
        level=0, name="Normal",
        shock_frequency_mult=1.0, shock_severity_mult=1.0,
        grading_strictness=1.0, citizen_trust_decay_mult=1.0,
        market_aggressiveness=0.3,
    ),
    DifficultyConfig(
        level=1, name="Challenging",
        shock_frequency_mult=1.3, shock_severity_mult=1.1,
        grading_strictness=1.1, citizen_trust_decay_mult=1.2,
        citizen_unrest_sensitivity=1.15,
        market_aggressiveness=0.45,
    ),
    DifficultyConfig(
        level=2, name="Hard",
        shock_frequency_mult=1.6, shock_severity_mult=1.25,
        grading_strictness=1.2, citizen_trust_decay_mult=1.4,
        citizen_unrest_sensitivity=1.3,
        capital_flight_sensitivity=1.2,
        initial_budget_penalty=200.0,
        market_aggressiveness=0.6,
        max_steps_reduction=2,
    ),
    DifficultyConfig(
        level=3, name="Expert+",
        shock_frequency_mult=2.0, shock_severity_mult=1.4,
        shock_duration_bonus=1,
        grading_strictness=1.35, citizen_trust_decay_mult=1.6,
        citizen_unrest_sensitivity=1.5,
        capital_flight_sensitivity=1.4,
        initial_budget_penalty=500.0,
        initial_inflation_add=0.02,
        market_aggressiveness=0.75,
        max_steps_reduction=3,
    ),
    DifficultyConfig(
        level=4, name="Nightmare",
        shock_frequency_mult=2.5, shock_severity_mult=1.6,
        shock_duration_bonus=2,
        grading_strictness=1.5, citizen_trust_decay_mult=2.0,
        citizen_unrest_sensitivity=1.8,
        capital_flight_sensitivity=1.6,
        initial_budget_penalty=800.0,
        initial_inflation_add=0.04,
        initial_unemployment_add=0.03,
        market_aggressiveness=0.9,
        max_steps_reduction=5,
    ),
]


class AdaptiveCurriculum:
    """
    Adaptive difficulty manager that automatically scales challenge
    based on agent performance.

    Usage:
        curriculum = AdaptiveCurriculum()
        difficulty = curriculum.get_difficulty()

        # After each episode:
        curriculum.update(score=0.75, task_id="task1_stability")
        new_difficulty = curriculum.get_difficulty()
    """

    def __init__(
        self,
        promotion_threshold: float = 0.70,
        demotion_threshold: float = 0.35,
        window_size: int = 5,
        max_level: int = 4,
    ):
        """
        Args:
            promotion_threshold: Score above which difficulty increases
            demotion_threshold: Score below which difficulty decreases
            window_size: Number of recent episodes to average over
            max_level: Maximum difficulty level
        """
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.window_size = window_size
        self.max_level = min(max_level, len(DIFFICULTY_LEVELS) - 1)

        self.current_level = 0
        self.score_history: list[float] = []
        self.level_history: list[int] = [0]
        self.promotions = 0
        self.demotions = 0
        self.episodes_at_level: dict[int, int] = {0: 0}

    def get_difficulty(self) -> DifficultyConfig:
        """Get the current difficulty configuration."""
        return DIFFICULTY_LEVELS[self.current_level]

    def update(self, score: float, task_id: str = "") -> dict:
        """
        Update curriculum based on episode score.
        Returns status dict with any level changes.
        """
        self.score_history.append(score)
        self.episodes_at_level[self.current_level] = (
            self.episodes_at_level.get(self.current_level, 0) + 1
        )

        result = {
            "score": score,
            "current_level": self.current_level,
            "level_changed": False,
            "direction": None,
        }

        # Need at least window_size episodes before considering promotion
        if len(self.score_history) < self.window_size:
            return result

        recent_avg = np.mean(self.score_history[-self.window_size:])
        result["recent_avg"] = float(recent_avg)

        # ── Promotion check ──────────────────────────────────────────────
        if recent_avg >= self.promotion_threshold and self.current_level < self.max_level:
            self.current_level += 1
            self.promotions += 1
            result["level_changed"] = True
            result["direction"] = "promoted"
            result["new_level"] = self.current_level
            result["message"] = (
                f"Level UP! {DIFFICULTY_LEVELS[self.current_level - 1].name} -> "
                f"{DIFFICULTY_LEVELS[self.current_level].name} "
                f"(avg score: {recent_avg:.3f} >= {self.promotion_threshold})"
            )

        # ── Demotion check ───────────────────────────────────────────────
        elif recent_avg < self.demotion_threshold and self.current_level > 0:
            self.current_level -= 1
            self.demotions += 1
            result["level_changed"] = True
            result["direction"] = "demoted"
            result["new_level"] = self.current_level
            result["message"] = (
                f"Level DOWN! {DIFFICULTY_LEVELS[self.current_level + 1].name} -> "
                f"{DIFFICULTY_LEVELS[self.current_level].name} "
                f"(avg score: {recent_avg:.3f} < {self.demotion_threshold})"
            )

        self.level_history.append(self.current_level)
        result["current_level"] = self.current_level
        return result

    def get_stats(self) -> dict:
        """Get curriculum statistics."""
        return {
            "current_level": self.current_level,
            "current_difficulty": DIFFICULTY_LEVELS[self.current_level].name,
            "total_episodes": len(self.score_history),
            "promotions": self.promotions,
            "demotions": self.demotions,
            "recent_avg": float(np.mean(self.score_history[-self.window_size:]))
                          if len(self.score_history) >= self.window_size else None,
            "all_time_avg": float(np.mean(self.score_history))
                            if self.score_history else 0.0,
            "best_score": float(max(self.score_history)) if self.score_history else 0.0,
            "episodes_per_level": dict(self.episodes_at_level),
            "level_progression": self.level_history,
        }

    def describe(self) -> str:
        """Human-readable description of current curriculum state."""
        diff = DIFFICULTY_LEVELS[self.current_level]
        stats = self.get_stats()
        return (
            f"Curriculum: {diff.describe()}\n"
            f"Episodes: {stats['total_episodes']} | "
            f"Promotions: {stats['promotions']} | "
            f"Demotions: {stats['demotions']} | "
            f"Best: {stats['best_score']:.3f}"
        )
