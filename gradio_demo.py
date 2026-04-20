"""
gradio_demo.py — Interactive Gradio Demo (#7)
═════════════════════════════════════════════
A live interactive demo where judges can:
  1. Select a task and click "Step" to watch the economy evolve
  2. See the agent's decisions with theory-grounded explanations
  3. View real-time dashboard updates
  4. Compare different agent strategies side-by-side

Usage:
    pip install gradio
    python gradio_demo.py
    # Opens at http://localhost:7861
"""

import os
import sys
import json
import base64
import numpy as np
import io

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from env.openenv_wrapper import SocialContractOpenEnv
from env.pydantic_models import PolicyAction
from graders.graders import run_grader
from demo import smart_policy
from explain_policy import explain_action, detect_strategy_phase

try:
    import gradio as gr
except ImportError:
    print("Install gradio: pip install gradio")
    sys.exit(1)


# ── Global State ─────────────────────────────────────────────────────────────
class DemoState:
    def __init__(self):
        self.env = None
        self.obs = None
        self.history = []
        self.step_count = 0
        self.task_id = "task1_stability"
        self.done = False

state = DemoState()


def reset_env(task_id, seed):
    """Reset the environment with a new task."""
    seed = int(seed) if seed else 42
    state.task_id = task_id
    state.env = SocialContractOpenEnv(task_id, seed=seed)
    state.obs = state.env.reset()
    state.history = []
    state.step_count = 0
    state.done = False

    status = (
        f"Environment reset! Task: {task_id}\n"
        f"Difficulty: {state.env.task_cfg['difficulty']}\n"
        f"Max steps: {state.env.task_cfg['max_steps']}\n\n"
        f"Initial state:\n"
        f"  GDP: {state.obs.gdp:,.1f}\n"
        f"  Gini: {state.obs.gini:.3f}\n"
        f"  Inflation: {state.obs.inflation:.3f}\n"
        f"  Unemployment: {state.obs.unemployment:.2f}\n"
        f"  Budget: {state.obs.gov_budget:,.0f}"
    )
    return status, generate_dashboard_image(), "Ready - click 'Step' to advance"


def step_env():
    """Take one step using the smart policy agent."""
    if state.env is None:
        return "Reset the environment first!", None, "Not initialized"

    if state.done:
        grade = run_grader(state.task_id, state.env._history)
        return (
            f"Episode finished!\n\nFinal Score: {grade['score']:.4f}\n"
            f"Verdict: {grade['verdict']}\n\n"
            f"Breakdown:\n{json.dumps(grade.get('breakdown', {}), indent=2)}",
            generate_dashboard_image(),
            f"Done! Score: {grade['score']:.4f}"
        )

    action = smart_policy(state.obs)
    phase = detect_strategy_phase(
        state.obs, state.task_id, state.obs.step,
        state.env.task_cfg["max_steps"]
    )
    explanations = explain_action(action, state.obs)

    state.obs = state.env.step(action)
    state.step_count += 1
    state.done = state.obs.done
    state.history.append(state.obs.metadata)

    # Build explanation text
    explanation_text = (
        f"Step {state.step_count} / {state.env.task_cfg['max_steps']}\n"
        f"Phase: {phase}\n\n"
        f"--- Current State ---\n"
        f"GDP: {state.obs.gdp:,.1f} | Gini: {state.obs.gini:.3f}\n"
        f"Inflation: {state.obs.inflation:.3f} | Unemployment: {state.obs.unemployment:.2f}\n"
        f"Budget: {state.obs.gov_budget:,.0f} | Unrest: {state.obs.unrest:.3f}\n"
        f"Consumer Confidence: {state.obs.consumer_confidence:.2f}\n\n"
        f"--- Agent Decision ---\n"
        f"Tax: {action.tax_delta:+.3f} | UBI: {action.ubi_delta:+.1f} | "
        f"PG: {action.public_good_delta:+.3f}\n"
        f"Rate: {action.interest_rate_delta:+.3f} | "
        f"Stimulus: {action.stimulus_package:.0f} | "
        f"QE: {action.money_supply_delta:+.0f}\n"
        f"Min Wage: {action.minimum_wage_delta:+.1f}\n\n"
        f"--- Policy Reasoning ---\n"
    )
    for exp in explanations:
        explanation_text += f"  * {exp}\n"

    explanation_text += (
        f"\n--- Reward ---\n"
        f"Step Reward: {state.obs.reward:.4f} | "
        f"Task Progress: {state.obs.metadata.get('reward_breakdown', {}).get('task_progress', 0):.4f}"
    )

    if state.obs.done:
        grade = run_grader(state.task_id, state.env._history)
        explanation_text += (
            f"\n\n=== EPISODE COMPLETE ===\n"
            f"Final Score: {grade['score']:.4f}\n"
            f"Verdict: {grade['verdict']}"
        )

    status = f"Step {state.step_count} | Reward: {state.obs.reward:.3f}" + (" | DONE" if state.obs.done else "")

    return explanation_text, generate_dashboard_image(), status


def run_full_episode():
    """Run the entire episode automatically."""
    if state.env is None:
        return "Reset the environment first!", None, "Not initialized"

    while not state.done:
        step_env()

    grade = run_grader(state.task_id, state.env._history)
    text = (
        f"Episode complete in {state.step_count} steps.\n\n"
        f"Final Score: {grade['score']:.4f}\n"
        f"Verdict: {grade['verdict']}\n\n"
        f"Breakdown:\n{json.dumps(grade.get('breakdown', {}), indent=2)}"
    )
    return text, generate_dashboard_image(), f"Done! Score: {grade['score']:.4f}"


def generate_dashboard_image():
    """Generate a live dashboard from the current history."""
    if not state.env or not state.env._history:
        # Return a placeholder
        fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        fig.patch.set_facecolor('#1a1a2e')
        ax.set_facecolor('#1a1a2e')
        ax.text(0.5, 0.5, "Reset the environment to begin",
                ha='center', va='center', fontsize=16, color='white')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                    facecolor='#1a1a2e')
        plt.close(fig)
        buf.seek(0)

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        tmp.write(buf.getvalue())
        tmp.close()
        return tmp.name

    history = state.env._history
    steps = [h["step"] for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a2e')
    fig.suptitle(f"SocialContract-v0 - {state.task_id} (Step {state.step_count})",
                 fontsize=14, fontweight="bold", color="white")

    metrics = [
        ("GDP", [h["gdp"] for h in history], '#4ECDC4', None),
        ("Gini (Inequality)", [h["gini"] for h in history], '#FF6B6B', 0.48),
        ("Gov Budget", [h["gov_budget"] for h in history], '#45B7D1', 0),
        ("Inflation", [h["inflation"] for h in history], '#96CEB4', 0.04),
        ("Unemployment", [h["unemployment"] for h in history], '#FFEAA7', 0.08),
        ("Reward", [h["reward"] for h in history], '#DDA0DD', None),
    ]

    for i, (title, values, color, target) in enumerate(metrics):
        ax = axes[i // 3][i % 3]
        ax.set_facecolor('#1a1a2e')
        ax.plot(steps, values, color=color, linewidth=2)
        ax.fill_between(steps, values, alpha=0.15, color=color)
        if target is not None:
            ax.axhline(y=target, color='orange', linestyle='--',
                       alpha=0.5, linewidth=1)
        ax.set_title(title, fontsize=10, fontweight="bold", color="white")
        ax.grid(True, alpha=0.15, color='#2d2d4a')
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#2d2d4a')

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight',
                facecolor='#1a1a2e')
    plt.close(fig)
    buf.seek(0)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    tmp.write(buf.getvalue())
    tmp.close()
    return tmp.name


# ── Gradio Interface ─────────────────────────────────────────────────────────

def build_interface():
    with gr.Blocks(
        title="SocialContract-v0 Interactive Demo",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="green",
        ),
        css="""
        .gradio-container { max-width: 1200px !important; }
        .explanation-box { font-family: monospace; font-size: 12px; }
        """
    ) as demo:

        gr.Markdown("""
        # SocialContract-v0 - Interactive Policy Advisory Demo

        **Watch an AI economic advisor manage a simulated economy in real-time.**
        Select a task, reset the environment, then step through the episode
        to see theory-grounded policy decisions with explanations.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                task_dropdown = gr.Dropdown(
                    choices=[
                        ("T1: Maintain Stability (Easy)", "task1_stability"),
                        ("T2: Recession Recovery (Medium)", "task2_recession"),
                        ("T3: Inequality Crisis (Hard)", "task3_crisis"),
                        ("T4: Stagflation Crisis (Expert)", "task4_stagflation"),
                        ("T5: Pandemic Response (Expert)", "task5_pandemic"),
                    ],
                    value="task4_stagflation",
                    label="Select Task",
                )
                seed_input = gr.Number(value=42, label="Random Seed")

                with gr.Row():
                    reset_btn = gr.Button("Reset", variant="secondary")
                    step_btn = gr.Button("Step", variant="primary")
                    run_btn = gr.Button("Run Full Episode", variant="stop")

                status_text = gr.Textbox(label="Status", value="Select a task and click Reset")

            with gr.Column(scale=2):
                dashboard_image = gr.Image(label="Live Dashboard", type="filepath")

        with gr.Row():
            explanation_box = gr.Textbox(
                label="Agent Decision & Explanation",
                lines=18,
                elem_classes=["explanation-box"],
            )

        # Wire up events
        reset_btn.click(
            fn=reset_env,
            inputs=[task_dropdown, seed_input],
            outputs=[explanation_box, dashboard_image, status_text],
        )
        step_btn.click(
            fn=step_env,
            outputs=[explanation_box, dashboard_image, status_text],
        )
        run_btn.click(
            fn=run_full_episode,
            outputs=[explanation_box, dashboard_image, status_text],
        )

    return demo


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    demo = build_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
