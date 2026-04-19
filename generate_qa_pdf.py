"""Generate Judge Q&A Prep PDF with term explanations."""
import os
from fpdf import FPDF

def safe(text):
    """Replace Unicode chars with ASCII equivalents."""
    return (text
        .replace('\u2014', ' -- ')
        .replace('\u2013', '-')
        .replace('\u2019', "'")
        .replace('\u2018', "'")
        .replace('\u201c', '"')
        .replace('\u201d', '"')
        .replace('\u2192', '->')
        .replace('\u2190', '<-')
        .replace('\u2264', '<=')
        .replace('\u2265', '>=')
        .replace('\u00b1', '+/-')
        .replace('\u2026', '...')
        .replace('\u2022', '*')
        .replace('\u2248', '~=')
        .replace('\u03b3', 'gamma')
        .replace('\u03c0', 'pi')
        .replace('\u03b8', 'theta')
        .replace('\u03b1', 'alpha')
        .replace('\u03b2', 'beta')
    )

class QAPDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 7, safe('SocialContract-v0 -- Judge Q&A Preparation Guide'), new_x="RIGHT", new_y="TOP", align='C')
        self.ln(9)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, safe(f'Page {self.page_no()}/{{nb}}'), new_x="RIGHT", new_y="TOP", align='C')

    def title_block(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 30, 100)
        self.cell(0, 12, safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 30, 100)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def question_block(self, num, question):
        if self.get_y() > 240:
            self.add_page()
        self.set_fill_color(30, 30, 100)
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, safe(f'  Q{num}: {question}'), new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(3)

    def answer_block(self, text):
        self.set_fill_color(235, 245, 235)
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(30, 100, 30)
        self.cell(20, 6, '  ANSWER:', new_x="RIGHT", new_y="TOP")
        self.set_font('Helvetica', '', 10)
        self.set_text_color(30, 60, 30)
        self.ln(8)
        y_start = self.get_y()
        self.set_x(12)
        self.multi_cell(186, 5.5, safe(text))
        y_end = self.get_y()
        self.set_fill_color(30, 130, 30)
        self.rect(10, y_start - 2, 1.5, y_end - y_start + 4, 'F')
        self.ln(2)

    def term_block(self, term, explanation):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(30, 30, 100)
        self.set_fill_color(240, 240, 252)
        self.cell(45, 6, safe(f'  {term}'), new_x="RIGHT", new_y="TOP", fill=True)
        self.set_font('Helvetica', '', 9)
        self.set_text_color(50, 50, 50)
        self.multi_cell(143, 6, safe(explanation), fill=True)
        self.ln(0.5)

    def section_label(self, text):
        self.set_font('Helvetica', 'B', 10)
        self.set_text_color(150, 80, 0)
        self.cell(0, 7, safe(f'Key Terms Used:'), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def tip_box(self, text):
        if self.get_y() > 255:
            self.add_page()
        self.set_fill_color(255, 248, 220)
        w = 190
        self.set_font('Helvetica', '', 9)
        txt = safe(text)
        lines = self.multi_cell(w - 10, 5, txt, dry_run=True, output="LINES")
        h = len(lines) * 5 + 8
        y_start = self.get_y()
        self.rect(10, y_start, w, h, 'F')
        self.set_fill_color(200, 150, 0)
        self.rect(10, y_start, 2, h, 'F')
        self.set_text_color(130, 80, 0)
        self.set_font('Helvetica', 'B', 9)
        self.set_xy(15, y_start + 2)
        self.cell(20, 5, 'TIP:', new_x="RIGHT", new_y="TOP")
        self.set_font('Helvetica', '', 9)
        self.multi_cell(w - 25, 5, txt)
        self.set_y(y_start + h + 3)

    def danger_box(self, text):
        if self.get_y() > 255:
            self.add_page()
        self.set_fill_color(255, 230, 230)
        w = 190
        txt = safe(text)
        self.set_font('Helvetica', '', 9)
        lines = self.multi_cell(w - 10, 5, txt, dry_run=True, output="LINES")
        h = len(lines) * 5 + 8
        y_start = self.get_y()
        self.rect(10, y_start, w, h, 'F')
        self.set_fill_color(200, 30, 30)
        self.rect(10, y_start, 2, h, 'F')
        self.set_text_color(180, 30, 30)
        self.set_font('Helvetica', 'B', 9)
        self.set_xy(15, y_start + 2)
        self.cell(30, 5, 'AVOID:', new_x="RIGHT", new_y="TOP")
        self.set_font('Helvetica', '', 9)
        self.multi_cell(w - 35, 5, txt)
        self.set_y(y_start + h + 3)

    def body(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, safe(text))
        self.ln(2)

    def heading(self, text):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(50, 50, 130)
        self.cell(0, 10, safe(text), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)


pdf = QAPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ===== TITLE PAGE =====
pdf.add_page()
pdf.ln(35)
pdf.set_font('Helvetica', 'B', 26)
pdf.set_text_color(30, 30, 100)
pdf.cell(0, 14, 'Judge Q&A Preparation', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, '25 Expected Questions with Answers', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 8, '& Key Term Explanations', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.ln(8)
pdf.set_draw_color(30, 30, 100)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)
pdf.set_font('Helvetica', '', 11)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 7, 'SocialContract-v0 Team', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 7, 'Meta Scaler Open Environment Hackathon 2025', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.ln(15)
pdf.set_font('Helvetica', 'B', 10)
pdf.set_text_color(30, 30, 100)
pdf.cell(0, 7, 'How to use this guide:', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 7, 'Each question has: (1) The answer to give, (2) Key terms explained', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 7, 'Practice each answer out loud. Keep it under 30 seconds.', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 7, 'Questions are grouped: RL Basics, Training, Environment, Multi-Agent, Results', new_x="LMARGIN", new_y="NEXT", align='C')

# ═══════════════════════════════════════════════════════════
# SECTION 1: RL AND TRAINING QUESTIONS
# ═══════════════════════════════════════════════════════════
pdf.add_page()
pdf.title_block('Section 1: RL & Training Questions')

# Q1
pdf.question_block(1, '"What training algorithm do you use?"')
pdf.answer_block(
    '"We use GRPO -- Group Relative Policy Optimization -- from the DeepSeek-R1 paper. '
    'For each economic scenario, the LLM generates 4 different policy responses. Our '
    'environment scores each one using the grader. GRPO computes relative advantages '
    'within each group and updates the model to produce more high-scoring responses. '
    'Unlike PPO, GRPO does not need a separate critic network, which makes it simpler '
    'and cheaper to run."'
)
pdf.section_label('')
pdf.term_block('GRPO', 'Group Relative Policy Optimization. Generates N completions per prompt, scores all of them, then ranks them relative to each other. "Best in group = increase probability, worst = decrease." Invented by DeepSeek (2024).')
pdf.term_block('Critic network', 'A second neural network in PPO that estimates "how good is this state?" (the value function V(s)). GRPO eliminates this by comparing completions directly against each other instead.')
pdf.term_block('Relative advantage', 'How much better/worse a completion is compared to the GROUP AVERAGE. Formula: advantage_i = (reward_i - mean) / std. Positive = better than average, negative = worse.')
pdf.tip_box('If they ask "why not PPO?" say: PPO needs a critic network (extra cost). GRPO achieves similar results without one. DeepSeek-R1 proved GRPO works for LLM training at scale.')

# Q2
pdf.add_page()
pdf.question_block(2, '"What is the reward function?"')
pdf.answer_block(
    '"Multi-objective: 35% GDP growth, 25% equality (1 minus Gini), 15% citizen satisfaction, '
    'minus penalties for: 12% social unrest, 8% budget deficit, 5% excessive taxes, and 8% '
    'policy volatility. The volatility penalty is critical -- it forces the agent to learn '
    'consistent strategies rather than flip-flopping between opposites each quarter."'
)
pdf.section_label('')
pdf.term_block('Multi-objective', 'The reward has MULTIPLE goals that conflict. You cannot maximize GDP alone (that causes inflation). You cannot maximize equality alone (that kills growth). The agent must find the BALANCE.')
pdf.term_block('Gini coefficient', 'A number from 0 to 1 measuring inequality. 0 = everyone has equal wealth. 1 = one person has everything. Real-world: USA ~0.41, Sweden ~0.28, South Africa ~0.63.')
pdf.term_block('Volatility penalty', 'Punishes the agent for making big reversals. If you raise taxes one quarter then cut them the next, you get penalized. This encourages CONSISTENT strategy, like real economic advisors.')
pdf.term_block('Policy flip-flopping', 'Changing direction rapidly (e.g., raise rates then cut rates). In real economics, this destroys market confidence. In our env, it erodes citizen trust and triggers volatility penalty.')

# Q3
pdf.question_block(3, '"How do you prevent reward hacking?"')
pdf.answer_block(
    '"Three mechanisms: First, the volatility penalty punishes erratic policy changes, so the '
    'agent cannot exploit one lever. Second, our phase-aware grader evaluates the ENTIRE '
    'trajectory -- not just the final state -- so you cannot cheat by optimizing the last step. '
    'Third, multi-seed evaluation across 5 different random seeds ensures the agent generalizes '
    'and is not memorizing a single scenario."'
)
pdf.section_label('')
pdf.term_block('Reward hacking', 'When the agent finds a loophole to get high reward without actually solving the task. Example: printing infinite money (QE=500) to boost GDP temporarily, even though it causes hyperinflation.')
pdf.term_block('Phase-aware grading', 'The grader does not just look at final GDP/Gini. It analyzes the SEQUENCE of actions. For stagflation, it checks: "Did the agent fight inflation first (Phase 1) THEN grow the economy (Phase 2)?"')
pdf.term_block('Multi-seed evaluation', 'Running the same agent on 5 different random environments (different shock sequences). If the agent only works on seed 42 but fails on seed 99, it has not truly learned -- it memorized.')
pdf.term_block('Trajectory', 'The complete sequence of (state, action, reward) from step 0 to the final step. Our grader analyzes the entire trajectory, not just a snapshot.')

# Q4
pdf.add_page()
pdf.question_block(4, '"What model do you use and why?"')
pdf.answer_block(
    '"Qwen 2.5 1.5B Instruct, fine-tuned with 4-bit QLoRA via Unsloth. We chose this model '
    'because: (1) it is small enough to train on a free Google Colab T4 GPU in under an hour, '
    '(2) it supports structured JSON output well, and (3) after GRPO training, it outperforms '
    'GPT-4o-mini zero-shot on our hardest tasks -- 0.788 versus 0.744. This proves our '
    'environment genuinely teaches economic reasoning to a small model."'
)
pdf.section_label('')
pdf.term_block('Qwen 2.5 1.5B', 'A 1.5 billion parameter language model by Alibaba. "Instruct" means it is already fine-tuned to follow instructions. We add RL on top of this base.')
pdf.term_block('4-bit QLoRA', 'Quantized Low-Rank Adaptation. The base model is compressed to 4-bit precision (750MB instead of 6GB), and we add tiny trainable adapters. Trains only 0.13% of parameters.')
pdf.term_block('Unsloth', 'A library that makes LoRA training 2x faster with custom CUDA kernels and memory optimizations. Key enabler for free T4 GPU training.')
pdf.term_block('Zero-shot', 'Testing a model without any task-specific training. GPT-4o-mini "zero-shot" means we gave it the economic report and asked for policy -- no fine-tuning on our env. Our 1.5B model AFTER training beats this.')
pdf.term_block('T4 GPU', 'NVIDIA Tesla T4: 16GB memory, available free on Google Colab. The fact that we can train on this shows our approach is accessible and reproducible.')

# Q5
pdf.question_block(5, '"How does LoRA work?"')
pdf.answer_block(
    '"Instead of updating all 1.5 billion parameters, LoRA adds small adapter matrices to '
    'each layer. Each adapter has about 131,000 parameters -- tiny compared to the 16.7 '
    'million in the original layer. We freeze the original model and only train the adapters. '
    'Total trainable parameters: about 2 million, which is 0.13% of the full model. This is '
    'what makes training feasible on a free GPU."'
)
pdf.section_label('')
pdf.term_block('LoRA', 'Low-Rank Adaptation. Adds two small matrices A (4096x16) and B (16x4096) to each layer. Output = original_weight * x + A * B * x. Only A and B are trained.')
pdf.term_block('Rank (r)', 'The inner dimension of the LoRA adapters. r=16 means each adapter is a 16-dimensional bottleneck. Higher rank = more capacity but slower training. r=16 is the standard sweet spot.')
pdf.term_block('Frozen model', 'The original 1.5B parameters are "frozen" -- they do not change during training. Only the small LoRA adapters are updated. This prevents catastrophic forgetting of language ability.')
pdf.term_block('Catastrophic forgetting', 'When fine-tuning destroys the model\'s original abilities. Without LoRA, training on economics could make the model forget how to write English. LoRA prevents this.')

# ═══════════════════════════════════════════════════════════
# SECTION 2: ENVIRONMENT QUESTIONS
# ═══════════════════════════════════════════════════════════
pdf.add_page()
pdf.title_block('Section 2: Environment Questions')

# Q6
pdf.question_block(6, '"Walk me through the environment."')
pdf.answer_block(
    '"The agent is an economic policy advisor managing a simulated country. Each quarter, it '
    'receives a report with 28 economic indicators -- GDP, Gini, inflation, unemployment, '
    'budget, consumer confidence, etc. The agent adjusts 8 policy levers simultaneously: '
    'tax rate, UBI, interest rates, public spending, stimulus, money supply (QE), tariffs, '
    'and minimum wage. The economy has 100 citizens across 4 wealth classes who react '
    'independently -- they can evade taxes, go on strike, or flee with capital. After 20-40 '
    'steps, a phase-aware grader scores the full trajectory from 0 to 1."'
)
pdf.section_label('')
pdf.term_block('28 observation fields', 'GDP, GDP delta, Gini, satisfaction, unrest, gov budget, tax rate, UBI, public goods, inflation, unemployment, interest rate, tariff, confidence, money supply, min wage, capital flight, strike, investment, and 4 per-class wealth values, plus task info.')
pdf.term_block('8 policy levers', 'All adjusted simultaneously each step. Not "pick one action" -- the agent outputs 8 continuous numbers. This is much harder than typical RL environments with discrete actions.')
pdf.term_block('4 wealth classes', 'Poor (50 citizens, high consumption), Middle (30, balanced), Wealthy (15, tax-sensitive), Ultra-rich (5, capital flight risk). Each class responds differently to policies.')
pdf.term_block('Quarter', 'One time step = one quarter (3 months) of economic policy. 20 steps = 5 years. 40 steps = 10 years.')

# Q7
pdf.question_block(7, '"What makes this environment hard for LLMs?"')
pdf.answer_block(
    '"Three things make this genuinely hard: First, it is high-dimensional -- 28 observations '
    'and 8 continuous action levers. The agent cannot just move left or right. Second, trust '
    'dynamics create temporal dependencies -- citizens remember past policies. If you flip-flop, '
    'trust erodes and citizens start evading taxes or going on strike. Third, emergent behaviors '
    'like capital flight, strikes, and tax evasion are not hard-coded rules -- they arise from '
    'citizen-level economic logic. The agent must learn to anticipate these second-order effects."'
)
pdf.section_label('')
pdf.term_block('High-dimensional', 'Many inputs and outputs. Most RL benchmarks have 4-10 actions. Yours has 8 continuous levers, each in a range. The combinatorial space is essentially infinite.')
pdf.term_block('Temporal dependency', 'Current outcomes depend on PAST actions. If you raised taxes last quarter, citizens remember and trust you less this quarter. The agent must think about sequences, not single steps.')
pdf.term_block('Emergent behavior', 'Complex behaviors that arise from simple rules. Tax evasion is not a switch -- it gradually increases as trust drops and tax rate rises. The agent never sees a rule saying "citizens evade taxes over 40%."')
pdf.term_block('Second-order effects', 'The consequence of a consequence. Raising taxes (1st order: more revenue) -> citizens evade more (2nd order: less revenue than expected) -> trust drops (3rd order: capital flight starts).')

# Q8
pdf.add_page()
pdf.question_block(8, '"Explain the grading system."')
pdf.answer_block(
    '"We have 5 task-specific graders, each scoring 0 to 1. They are PHASE-AWARE, meaning '
    'they analyze the full trajectory, not just the final state. For Task 4 (stagflation), '
    'the grader checks if the agent executed a 2-phase Volcker strategy -- first fight '
    'inflation with rate hikes and monetary tightening, THEN stimulate growth. It also applies '
    'volatility penalties for policy oscillation and simultaneous achievement gates -- you must '
    'hit multiple targets at the same time to score above 0.60."'
)
pdf.section_label('')
pdf.term_block('Phase-aware', 'The grader detects strategic phases in the trajectory. For stagflation: Phase 1 = inflation control (first 50% of steps), Phase 2 = growth recovery (last 50%). Score bonus if agent executes both phases correctly.')
pdf.term_block('Volcker strategy', 'Named after Paul Volcker, Fed Chair 1979-1987. He raised interest rates dramatically to kill inflation (causing short-term pain), then loosened policy to allow recovery. Our grader checks if the agent replicates this logic.')
pdf.term_block('Achievement gates', 'Multiple conditions that must be TRUE simultaneously. For Task 1: GDP growth > 0 AND Gini < 0.43 AND unrest < 0.15 AND budget > 0, all at the same time. Hitting just one is not enough.')
pdf.term_block('Trajectory scoring', 'Instead of scoring the final snapshot, the grader looks at the ENTIRE history. Did GDP trend upward? Did inflation decrease over time? Was the strategy consistent? This prevents "last-step optimization" tricks.')

# Q9
pdf.question_block(9, '"What are the 5 tasks?"')
pdf.answer_block(
    '"Task 1: Stability Maintenance (easy, 20 steps) -- keep a healthy economy stable. '
    'Task 2: Recession Recovery (medium, 30 steps) -- recover from a Greece-2010-style crash. '
    'Task 3: Inequality Crisis (hard, 30 steps) -- reduce extreme inequality like post-Soviet Russia. '
    'Task 4: Stagflation (expert, 35 steps) -- solve 1973-style simultaneous high inflation and unemployment. '
    'Task 5: Pandemic Response (expert, 30 steps) -- manage COVID-style economic shutdown with rolling waves. '
    'Each task is calibrated to real historical economic events."'
)
pdf.section_label('')
pdf.term_block('Stagflation', 'Simultaneous high inflation + high unemployment. Normally these are inversely related (Phillips Curve). Stagflation breaks normal economic tools because fighting inflation worsens unemployment and vice versa.')
pdf.term_block('Phillips Curve', 'The economic relationship between inflation and unemployment: usually when one goes up, the other goes down. Our environment implements this with the NAIRU (Natural Rate of Unemployment) at 4.5%.')
pdf.term_block('Calibrated', 'Our initial conditions match real data. Task 2 starts with the same GDP contraction, deficit, and unemployment as Greece 2010. Task 5 matches Q2-2020 OECD COVID averages.')

# ═══════════════════════════════════════════════════════════
# SECTION 3: MULTI-AGENT QUESTIONS
# ═══════════════════════════════════════════════════════════
pdf.add_page()
pdf.title_block('Section 3: Multi-Agent Questions')

# Q10
pdf.question_block(10, '"Is this really multi-agent?"')
pdf.answer_block(
    '"Yes. We have a market consortium of 3 independent agents: an institutional investor, a '
    'hedge fund, and a momentum trader. Each has different risk tolerance (aggressiveness 0.2, '
    '0.7, 0.5), different capital allocation strategies, and independent decision-making. The '
    'hedge fund can trigger speculative attacks when it detects policy weakness. The LLM must '
    'learn to maintain market confidence -- not just optimize GDP. The market agents are scripted '
    'but strategic, similar to how game environments use scripted opponents before self-play."'
)
pdf.section_label('')
pdf.term_block('Market consortium', 'A group of 3 market agents that collectively represent "the market." Their weighted responses determine investment flows, capital flight, and potential speculative attacks.')
pdf.term_block('Speculative attack', 'When speculators bet against a country\'s economy, causing a self-fulfilling crisis. Real example: George Soros vs Bank of England in 1992 (Black Wednesday). In our env, the hedge fund can do this.')
pdf.term_block('Scripted vs learning', 'Our market agents follow rules (scripted), not ML. Judges might challenge this. Answer: "Like game AI that uses expert heuristics. The multi-agent DYNAMICS are genuine -- the LLM must reason about market reactions."')
pdf.danger_box('If a judge says "these are just NPCs, not real agents" -- respond: "The market agents have independent state, make autonomous portfolio decisions, and create genuine strategic interaction. The LLM observes their effects indirectly (partial observability) and must infer their behavior. This is standard in multi-agent RL literature -- opponents do not have to be learning agents."')

# Q11
pdf.question_block(11, '"How do speculative attacks work?"')
pdf.answer_block(
    '"When the hedge fund detects: government confidence below 0.3, inflation above 6%, a large '
    'deficit, and social unrest -- it launches an attack based on Diamond-Dybvig coordination '
    'failure theory. Capital flees domestic assets, GDP drops by up to 12%, unrest spikes, and '
    'confidence drops further -- creating a vicious cycle. The attack has a 5-step cooldown so '
    'it cannot happen every turn. The LLM must learn to PREVENT attacks by maintaining policy '
    'credibility and economic stability."'
)
pdf.section_label('')
pdf.term_block('Diamond-Dybvig', 'A 1983 economics paper explaining bank runs and coordination failures. When enough investors lose confidence simultaneously, their withdrawals CAUSE the crisis they feared. Self-fulfilling prophecy.')
pdf.term_block('Coordination failure', 'When individual rational decisions lead to collectively bad outcomes. Each investor rationally fleeing = everyone loses. Our market agents can trigger this cascade.')
pdf.term_block('Policy credibility', 'How much the market believes the government will stick to its strategy. Consistent policies build credibility. Flip-flopping destroys it. Based on Barro-Gordon (1983) theory.')

# ═══════════════════════════════════════════════════════════
# SECTION 4: RESULTS & DEMO QUESTIONS
# ═══════════════════════════════════════════════════════════
pdf.add_page()
pdf.title_block('Section 4: Results & Demo Questions')

# Q12
pdf.question_block(12, '"Show me the training improvement."')
pdf.answer_block(
    '"Three pieces of evidence: First, our reward curves show the GRPO-trained model\'s mean '
    'reward climbing from 0.30 (random baseline) to 0.79 over 200 training steps. Second, our '
    'before/after behavioral comparison shows the untrained model panicking and crashing the '
    'economy, while the trained model executes a clean 2-phase Volcker strategy. Third, our '
    'benchmark table shows the trained 1.5B model scoring 0.788 -- beating GPT-4o-mini at '
    '0.744 on expert tasks."'
)
pdf.section_label('')
pdf.term_block('Reward curve', 'A plot of reward (Y-axis) vs training steps (X-axis). Should go UP over time, showing the agent is learning. Our curve goes from 0.30 to 0.79.')
pdf.term_block('Before/after comparison', 'Running the same economic scenario with the model before training (random-like behavior) vs after training (strategic behavior). Shows qualitative learning.')
pdf.term_block('Benchmark table', 'Comparing multiple agents across all tasks. Random=0.290, Heuristic=0.656, GPT-4o-mini=0.744, GRPO-trained=0.788. The ranking proves our training works.')

# Q13
pdf.question_block(13, '"How reproducible is this?"')
pdf.answer_block(
    '"Fully reproducible in 3 commands: pip install requirements, python validate.py (37/37 checks), '
    'python demo.py (benchmark with scores). The Colab notebook runs end-to-end training in '
    'under an hour on a free T4 GPU. The environment is deterministic given a seed, and we test '
    'across 5 seeds. Everything is OpenEnv-compliant and passes all 37 validation criteria."'
)
pdf.section_label('')
pdf.term_block('OpenEnv-compliant', 'Follows the OpenEnv specification: typed Pydantic models for observations/actions, reset/step/state API, deterministic graders, proper openenv.yaml, Dockerfile, and inference script.')
pdf.term_block('Deterministic', 'Given the same seed and actions, the environment produces the exact same results every time. No randomness in reproduction. Only the shocks use the seeded RNG.')
pdf.term_block('37 validation checks', 'Our validate.py tests: YAML spec, Pydantic models, API interface, academic calibration (Phillips Curve, Okun\'s Law), grader bounds, inference script, Dockerfile, and FastAPI endpoints.')

# Q14
pdf.add_page()
pdf.question_block(14, '"What is the curriculum learning?"')
pdf.answer_block(
    '"We implement Automatic Domain Randomization with 5 difficulty levels. At level 0 '
    '(Normal), the environment runs standard parameters. At level 4 (Nightmare), shocks are '
    '2.5x more frequent, initial inflation is +4%, unemployment is +3%, and the agent has 5 '
    'fewer steps to solve the task. The curriculum auto-promotes when the agent\'s rolling '
    'average score exceeds 0.70, and demotes below 0.35. This creates recursive skill '
    'amplification -- the agent drives its own capability growth."'
)
pdf.section_label('')
pdf.term_block('ADR', 'Automatic Domain Randomization. Originated from OpenAI\'s Rubik\'s Cube robot hand paper (2019). The environment automatically increases difficulty when the agent masters the current level.')
pdf.term_block('Rolling average', 'Average score over the last N episodes (N=5 in our config). Smooths out noise so a single bad episode does not cause demotion.')
pdf.term_block('Recursive skill amplification', 'The agent learns skill A -> environment gets harder -> agent learns skill B (which requires A) -> gets harder again. The difficulty recursively builds on previous capabilities. Addresses Theme #4.')

# Q15
pdf.question_block(15, '"What are the limitations?"')
pdf.answer_block(
    '"Three honest limitations: First, our simulation is simplified -- real economies are more '
    'complex with millions of actors. Second, our market agents follow rules rather than '
    'learning, so they do not adapt to the LLM\'s strategy over time. Third, we have not yet '
    'tested the full training pipeline on models larger than 1.5B -- larger models might benefit '
    'even more from our rich environment. These are natural next steps."'
)
pdf.tip_box('Judges respect honesty about limitations. It shows maturity and understanding. Never claim your project is perfect.')

# ═══════════════════════════════════════════════════════════
# SECTION 5: CURVEBALL QUESTIONS
# ═══════════════════════════════════════════════════════════
pdf.add_page()
pdf.title_block('Section 5: Curveball Questions')

# Q16
pdf.question_block(16, '"Why not use real economic data?"')
pdf.answer_block(
    '"Real economies do not have a reset button. You cannot run 1000 episodes of the US economy. '
    'Simulations let us train through thousands of diverse scenarios with different shocks and '
    'seeds. Our parameters are calibrated from OECD, IMF, and World Bank data -- the simulation '
    'MATCHES real-world elasticities. Sim-to-real transfer is a standard approach in RL, used '
    'by OpenAI for robotics and DeepMind for nuclear fusion control."'
)
pdf.section_label('')
pdf.term_block('Sim-to-real', 'Training in simulation, then deploying in the real world. Standard practice because simulation allows unlimited safe experimentation. Our calibration from real data maximizes transfer.')
pdf.term_block('Elasticities', 'How sensitive one economic variable is to another. Example: price elasticity of demand -- if prices rise 10%, demand might drop 15% (elasticity = 1.5). Our citizen classes have different elasticities matching real data.')

# Q17
pdf.question_block(17, '"How does this compare to other submissions?"')
pdf.answer_block(
    '"Our environment spans multiple themes: Theme 1 (multi-agent via market speculators), '
    'Theme 2 (long-horizon planning with 20-40 step episodes), Theme 3.1 (professional fiscal '
    'advisory task), and Theme 4 (curriculum self-improvement). Most environments target only '
    'one theme. Our observation and action spaces are among the highest-dimensional in any '
    'OpenEnv submission, and we have real empirical evidence of training improvement."'
)
pdf.danger_box('Never trash other teams. Focus on your own strengths. Judges will notice if you are negative about competitors.')

# Q18
pdf.question_block(18, '"Can this work for real policy decisions?"')
pdf.answer_block(
    '"Not directly -- this is a training environment, not a decision support system. But the '
    'SKILLS it trains are real: multi-objective optimization, long-horizon planning, phase-aware '
    'strategy, and robustness to shocks. These are exactly the cognitive skills that IMF and '
    'World Bank advisors use daily. A model trained here could be a useful starting point for '
    'real policy advisory tools with proper calibration and human oversight."'
)
pdf.tip_box('Frame it as "training capability, not deployment tool." This shows you understand the difference between research and production.')

# Q19
pdf.add_page()
pdf.question_block(19, '"What is the KL penalty and why does it matter?"')
pdf.answer_block(
    '"KL divergence measures how much the model has changed from its original version. During '
    'GRPO training, we add a KL penalty to the loss function to prevent the model from drifting '
    'too far. Without it, the model might get very good at economics but forget how to write '
    'coherent English or generate valid JSON. The KL penalty is like guardrails -- it lets the '
    'model improve at the task while staying within safe bounds."'
)
pdf.section_label('')
pdf.term_block('KL divergence', 'Kullback-Leibler divergence: measures the "distance" between two probability distributions. KL(new || old) = how different the new model is from the original. Higher = more different = more risky.')
pdf.term_block('Distribution drift', 'When the model\'s output distribution shifts so far from the original that it loses core capabilities. Also called "mode collapse" in some contexts.')
pdf.term_block('Loss function', 'The mathematical formula the optimizer minimizes during training. GRPO loss = -advantage * log_prob + beta * KL. The beta controls how strong the KL penalty is.')

# Q20
pdf.question_block(20, '"What is the observation space vs action space?"')
pdf.answer_block(
    '"The observation space has 28 continuous fields -- GDP, Gini, inflation, unemployment, etc. '
    'This is what the agent sees as a text report. The action space has 8 continuous levers, '
    'each with a specific range. For example, tax_delta is in [-0.10, +0.10], stimulus is in '
    '[0, 500]. The agent outputs all 8 simultaneously as a JSON object. This high dimensionality '
    'is what makes our environment genuinely challenging for LLMs."'
)
pdf.section_label('')
pdf.term_block('Observation space', 'All possible inputs the agent can receive. "28 continuous fields" means 28 real numbers, each in a range. This is TEXT for LLMs -- formatted as an economic report.')
pdf.term_block('Action space', 'All possible outputs the agent can produce. "8 continuous levers" = 8 real numbers. This is effectively infinite -- unlike "pick action A, B, or C" in simple envs.')
pdf.term_block('Continuous vs discrete', 'Discrete: finite choices (left, right, jump). Continuous: infinite choices (any number in a range). Continuous is MUCH harder because there are infinite possible actions.')

# Q21
pdf.add_page()
pdf.question_block(21, '"Explain the trust dynamics."')
pdf.answer_block(
    '"Each citizen class has a trust level that represents belief in government credibility. '
    'Trust increases with consistent, gradual policies and decreases with flip-flopping or '
    'extreme changes. Low trust causes: citizens work less, evade more taxes, and wealthy '
    'citizens start moving capital offshore. If trust drops below 0.6 for poor/middle '
    'citizens AND their satisfaction is below 0.25, they coordinate a STRIKE -- shutting '
    'down economic output. This is based on Barro-Gordon credibility theory from 1983."'
)
pdf.section_label('')
pdf.term_block('Barro-Gordon', 'A 1983 paper on how government policy credibility affects economic outcomes. Consistent policy builds reputation. Surprise policy changes may work short-term but destroy long-term credibility.')
pdf.term_block('Trust erosion', 'Trust decreases by 0.03-0.05 per flip-flop. Once lost, it recovers slowly (0.01 per consistent step). Trust is asymmetric -- easy to lose, hard to rebuild. Like real government credibility.')
pdf.term_block('Capital flight', 'When wealthy citizens move money out of the country due to low trust or high taxes. Reduces domestic investment and GDP. Ultra-rich citizens start fleeing when trust < 0.6 or tax > 50%.')
pdf.term_block('Strike mechanics', 'When poor/middle citizens are unsatisfied AND distrustful, they coordinate a strike (Olson 1965, collective action). This reduces GDP output by ~15% and spikes unrest.')

# Q22
pdf.question_block(22, '"How do you handle the stochastic nature?"')
pdf.answer_block(
    '"Multi-seed evaluation across 5 seeds (42, 99, 7, 256, 1337). Our smart heuristic has '
    'standard deviation of only 0.02 across seeds, showing the environment is stable but not '
    'trivial. The grader is fully deterministic given the same trajectory. Economic shocks add '
    'healthy variance that tests generalization. Our validation script confirms determinism."'
)
pdf.section_label('')
pdf.term_block('Stochastic', 'Involving randomness. Our environment has random shocks (9 types: oil crises, tech booms, pandemics, etc.) that test if the agent can handle unexpected events.')
pdf.term_block('Seed', 'A number that initializes the random number generator. Same seed = same sequence of random events. This makes experiments reproducible while still having randomness.')
pdf.term_block('Standard deviation', 'Measures spread of scores. std=0.02 means most scores fall within +/-0.02 of the mean. Low std = consistent agent. High std = unreliable agent.')

# ═══════════════════════════════════════════════════════════
# SECTION 6: TOUGH / ADVERSARIAL QUESTIONS
# ═══════════════════════════════════════════════════════════
pdf.add_page()
pdf.title_block('Section 6: Tough / Adversarial Questions')

# Q23
pdf.question_block(23, '"Is this just a toy simulation?"')
pdf.answer_block(
    '"No. We implement 15 published economic theories including the Phillips Curve, Okun\'s Law, '
    'Taylor Rule, Laffer Curve, and Barro-Gordon credibility dynamics. Our parameters are '
    'calibrated from OECD, IMF, and World Bank data. The ablation study shows removing features '
    'inflates scores -- the complexity is justified: without trust dynamics, scores jump 12 '
    'points. Without phase-aware grading, scores jump 17 points. The difficulty is real."'
)
pdf.section_label('')
pdf.term_block('Ablation study', 'Removing one feature at a time to measure its impact. If removing Feature X makes the task much easier, then Feature X is important. Our ablation proves trust dynamics and phase grading are necessary.')
pdf.term_block('Phillips Curve', 'Inverse relationship between inflation and unemployment. Implemented in citizens.py with NAIRU=4.5%. When unemployment drops below NAIRU, inflation rises (and vice versa).')
pdf.term_block('Okun\'s Law', 'A 1% increase in unemployment above NAIRU causes ~2% reduction in GDP. Implemented with beta=2.0 in our environment. Real-world calibrated.')
pdf.term_block('Laffer Curve', 'Tax revenue peaks at some interior tax rate. Raise taxes too high and revenue drops because people work less or evade. Our formula: Revenue = t * (1-t^2) * base.')

# Q24
pdf.question_block(24, '"Why should we pick this over other submissions?"')
pdf.answer_block(
    '"Three reasons: First, we cover 4 hackathon themes with a single coherent environment -- '
    'multi-agent, long-horizon planning, professional tasks, and self-improvement. Second, we have '
    'complete empirical evidence: training curves, before/after comparison, and multi-model '
    'benchmarks showing a 1.5B model beating GPT-4o-mini. Third, we pass all 37 OpenEnv '
    'validation checks, have a working Colab notebook, and the entire pipeline reproduces '
    'in under an hour on free hardware."'
)
pdf.tip_box('Focus on EVIDENCE. Judges have seen many well-designed environments. What separates top teams is: proof it works, clear communication, and reproducibility.')

# Q25
pdf.question_block(25, '"What would you do with more time/compute?"')
pdf.answer_block(
    '"Three things: First, enable self-play between the LLM policy advisor and a LEARNING '
    'market speculator -- true multi-agent training. Second, scale to larger models (7B, 14B) '
    'to see how performance scales with model size. Third, add more economic scenarios -- trade '
    'wars between two countries, currency crises, and climate transition policies. The framework '
    'is extensible."'
)
pdf.tip_box('This question tests vision. Show you have thought beyond the hackathon. Judges want to fund ambitious projects with clear next steps.')

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Judge_QA_Guide.pdf')
pdf.output(out_path)
print(f"PDF saved to: {out_path}")
print(f"Total pages: {pdf.page_no()}")
