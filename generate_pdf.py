"""Generate RL Crash Course PDF - ASCII safe version."""
import os
from fpdf import FPDF

def safe(text):
    """Replace Unicode chars with ASCII equivalents."""
    return (text
        .replace('\u2014', ' - ')   # em-dash
        .replace('\u2013', '-')     # en-dash
        .replace('\u2019', "'")     # right single quote
        .replace('\u2018', "'")     # left single quote
        .replace('\u201c', '"')     # left double quote
        .replace('\u201d', '"')     # right double quote
        .replace('\u2192', '->')    # right arrow
        .replace('\u2190', '<-')    # left arrow
        .replace('\u2264', '<=')    # less than or equal
        .replace('\u2265', '>=')    # greater than or equal
        .replace('\u03b3', 'gamma') # gamma
        .replace('\u03c0', 'pi')    # pi
        .replace('\u03b8', 'theta') # theta
        .replace('\u2248', '~=')    # approximately
        .replace('\u00b1', '+/-')   # plus minus
        .replace('\u2026', '...')   # ellipsis
        .replace('\u2022', '*')     # bullet
        .replace('\u2713', '[OK]')  # checkmark
        .replace('\u2717', '[X]')   # cross
        .replace('\u03b1', 'alpha') # alpha
        .replace('\u03b2', 'beta')  # beta
        .replace('\u2211', 'SUM')   # sigma
        .replace('\u03c3', 'sigma') # sigma lowercase
        .replace('\u2206', 'delta') # delta
    )

class RLCoursePDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, safe('SocialContract-v0 - RL Crash Course for Hackathon Team'), new_x="RIGHT", new_y="TOP", align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, safe(f'Page {self.page_no()}/{{nb}}'), new_x="RIGHT", new_y="TOP", align='C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 16)
        self.set_text_color(30, 30, 100)
        self.cell(0, 12, safe(title), new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(30, 30, 100)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def section_title(self, title):
        self.set_font('Helvetica', 'B', 13)
        self.set_text_color(50, 50, 130)
        self.cell(0, 10, safe(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def sub_section(self, title):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(70, 70, 70)
        self.cell(0, 8, safe(title), new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body_text(self, text):
        self.set_font('Helvetica', '', 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, safe(text))
        self.ln(2)

    def code_block(self, text):
        self.set_font('Courier', '', 8.5)
        self.set_fill_color(240, 240, 245)
        self.set_text_color(30, 30, 30)
        lines = safe(text).split('\n')
        height = len(lines) * 4.5 + 6
        if self.get_y() + height > 270:
            self.add_page()
        self.rect(10, self.get_y(), 190, height, 'F')
        self.ln(3)
        for line in lines:
            self.cell(4)
            self.cell(0, 4.5, line[:105], new_x="LMARGIN", new_y="NEXT")
        self.ln(3)

    def important_box(self, text, color='blue'):
        colors = {
            'blue': (220, 235, 255, 30, 80, 180),
            'green': (220, 245, 220, 30, 120, 30),
            'orange': (255, 240, 220, 200, 100, 0),
        }
        bg_r, bg_g, bg_b, t_r, t_g, t_b = colors.get(color, colors['blue'])
        self.set_fill_color(bg_r, bg_g, bg_b)
        w = 190
        self.set_font('Helvetica', '', 9)
        txt = safe(text)
        lines = self.multi_cell(w - 10, 5, txt, split_only=True)
        h = len(lines) * 5 + 8
        if self.get_y() + h > 270:
            self.add_page()
        y_start = self.get_y()
        self.rect(10, y_start, w, h, 'F')
        self.set_fill_color(t_r, t_g, t_b)
        self.rect(10, y_start, 2, h, 'F')
        self.set_text_color(t_r, t_g, t_b)
        self.set_xy(15, y_start + 3)
        self.multi_cell(w - 15, 5, txt)
        self.set_y(y_start + h + 3)


pdf = RLCoursePDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ===== TITLE PAGE =====
pdf.add_page()
pdf.ln(40)
pdf.set_font('Helvetica', 'B', 28)
pdf.set_text_color(30, 30, 100)
pdf.cell(0, 15, 'RL Crash Course', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.set_font('Helvetica', '', 14)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 10, 'Everything You Need to Know for the Hackathon', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.ln(10)
pdf.set_draw_color(30, 30, 100)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(10)
pdf.set_font('Helvetica', '', 12)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 8, 'For: SocialContract-v0 Team (3 Members)', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 8, 'Meta Scaler Open Environment Hackathon 2025', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.ln(15)
pdf.set_font('Helvetica', 'B', 11)
pdf.set_text_color(30, 30, 100)
pdf.cell(0, 8, 'Team Study Plan', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.set_font('Helvetica', '', 10)
pdf.set_text_color(60, 60, 60)
pdf.cell(0, 7, 'Member 1: Parts 1-3 (Core RL + Mapping to Project)', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 7, 'Member 2: Parts 4-6 (RLHF, GRPO, PPO, Training Pipeline)', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 7, 'Member 3: Parts 7-9 (Multi-Agent, Curriculum, Reward Design)', new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 7, 'ALL: Parts 10 + 12 (Judge Q&A + Quick Reference Card)', new_x="LMARGIN", new_y="NEXT", align='C')

# ===== PART 1 =====
pdf.add_page()
pdf.chapter_title('Part 1: The Absolute Basics')

pdf.section_title('What is Reinforcement Learning?')
pdf.body_text(
    'RL is how you teach a computer to make good decisions by trial and error. '
    'Think of training a dog: the dog does something, you give it a treat (reward) '
    'or say "no" (penalty). Over time, the dog learns which actions get treats. '
    'The dog does not need a textbook -- it learns from experience.'
)
pdf.body_text(
    'The RL loop: An AGENT (your LLM) observes the world, takes an ACTION, '
    'gets a REWARD, and sees the new STATE. This loop repeats until the episode ends.'
)
pdf.code_block(
    '  AGENT (LLM)  ---action--->  ENVIRONMENT (Economy)\n'
    '               <--reward----\n'
    '               <--state-----'
)

pdf.section_title('The 5 Core Concepts')

pdf.sub_section('1. Agent')
pdf.body_text(
    'The thing that makes decisions. In your project: the LLM (Qwen 1.5B or GPT-4o-mini). '
    'It reads an economic report and outputs a policy decision as JSON.'
)

pdf.sub_section('2. Environment')
pdf.body_text(
    'The world the agent interacts with. In your project: SocialContract-v0. '
    'It simulates 100 citizens across 4 wealth classes with real economic dynamics '
    '(trust, tax evasion, strikes, capital flight). Shocks happen randomly.'
)

pdf.sub_section('3. State / Observation')
pdf.body_text(
    'What the agent can see. In your project: 28 economic fields including GDP, Gini, '
    'inflation, unemployment, government budget, consumer confidence, etc. '
    '"Partially observable" means the agent cannot see everything -- it only sees '
    'aggregate statistics, not individual citizen trust levels.'
)

pdf.sub_section('4. Action')
pdf.body_text(
    'What the agent does. In your project: 8 simultaneous policy levers -- tax rate, UBI, '
    'interest rate, stimulus, money supply (QE), public goods, tariffs, and minimum wage. '
    'These are CONTINUOUS (real numbers in a range), not discrete (pick one option).'
)

pdf.sub_section('5. Reward')
pdf.body_text(
    'The score the agent gets after each action. In your project: 0.0 to 1.0. '
    'Composed of: 35% GDP growth + 25% equality + 15% satisfaction - 12% unrest '
    '- 8% deficit penalty - 8% volatility penalty. Multi-objective!'
)

pdf.important_box(
    'KEY INSIGHT: Your reward function is multi-objective. The agent cannot just maximize '
    'GDP -- it must also reduce inequality, keep citizens happy, avoid deficits, and '
    'maintain policy consistency. This forces genuine tradeoff reasoning.', 'blue'
)

# ===== PART 2 =====
pdf.add_page()
pdf.chapter_title('Part 2: Key RL Terms Glossary')

terms = [
    ('Episode', 'One complete game start to finish', 'One task run (20-40 steps)'),
    ('Step', 'One turn in the game', 'One quarter of economic policy'),
    ('Trajectory', 'Full sequence of states+actions', 'Complete economic history'),
    ('Horizon', 'How many steps agent plans', '20-40 steps (long-horizon!)'),
    ('Discount (gamma)', 'How much agent values future', 'Must plan ahead'),
    ('Return', 'Total accumulated reward', 'Sum of all step rewards'),
    ('Policy (pi)', 'Agent strategy: obs -> action', '"when inflation high, raise rates"'),
    ('Exploration', 'Trying new strategies', 'Temperature > 0 in LLM'),
    ('Exploitation', 'Using best known strategy', 'Temperature = 0 in LLM'),
    ('Sparse reward', 'Only feedback at end', 'Your grader = final score'),
    ('Dense reward', 'Feedback every step', 'Your per-step reward'),
    ('Action space', 'All possible actions', '8 continuous levers'),
    ('Reward shaping', 'Designing reward well', 'Your volatility penalty'),
    ('Curriculum', 'Easy first, then harder', 'Normal to Nightmare'),
    ('Multi-agent', 'Multiple decision-makers', 'LLM vs market speculators'),
    ('Baseline', 'Simple comparison agent', 'Random=0.29, Heuristic=0.66'),
    ('MDP/POMDP', 'Math model of RL problem', 'Yours is POMDP'),
    ('Value fn V(s)', 'How good is this state?', 'Expected future reward'),
    ('Q-fn Q(s,a)', 'How good is action a here?', 'Q value of policy choice'),
    ('Advantage', 'Better than average?', 'Q(s,a)-V(s). Used in GRPO!'),
    ('On-policy', 'Learn from own experience', 'PPO and GRPO'),
    ('Rollout', 'Running policy in env', 'Running LLM in your env'),
    ('Overfitting', 'Memorize, no generalize', 'Fixed by multi-seed eval'),
    ('Reward hacking', 'Exploit reward loopholes', 'Fixed by volatility pen.'),
    ('KL divergence', 'Dist. between 2 models', 'Prevents model drift'),
]

for term, desc, proj in terms:
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(30, 30, 100)
    pdf.cell(35, 5.5, safe(term), new_x="RIGHT", new_y="TOP")
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(40, 40, 40)
    pdf.cell(65, 5.5, safe(desc), new_x="RIGHT", new_y="TOP")
    pdf.set_font('Helvetica', 'I', 8.5)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(90, 5.5, safe(proj), new_x="LMARGIN", new_y="NEXT")

# ===== PART 3 =====
pdf.add_page()
pdf.chapter_title('Part 3: How RL Applies to LLMs')

pdf.section_title('Traditional RL vs LLM RL')
pdf.body_text(
    'Traditional RL (games, robots) trains small neural networks through millions of episodes. '
    'LLM RL trains huge language models through hundreds of episodes (much more expensive). '
    'Key difference: LLM actions are TEXT (generate JSON), not discrete moves (left/right).'
)

pdf.section_title('The LLM RL Training Pipeline (what train_grpo.py does)')
pdf.code_block(
    'TRAINING LOOP:\n'
    '  1. GENERATE PROMPT\n'
    '     - Reset environment, get observation\n'
    '     - Format as text prompt for the LLM\n'
    '\n'
    '  2. GENERATE COMPLETIONS (the LLM\'s "actions")\n'
    '     - LLM reads the prompt\n'
    '     - Generates N=4 different policy JSONs\n'
    '\n'
    '  3. EVALUATE (get rewards)\n'
    '     - Parse each JSON into a PolicyAction\n'
    '     - Run env.step() with each action\n'
    '     - Get grader score = reward signal\n'
    '\n'
    '  4. UPDATE MODEL (learn from rewards)\n'
    '     - Increase probability of high-reward completions\n'
    '     - Decrease probability of low-reward completions\n'
    '     - KL penalty prevents catastrophic change\n'
    '\n'
    '  5. REPEAT'
)

pdf.important_box(
    'KEY FOR JUDGES: The environment IS the reward function. Your run_grader() output '
    'directly becomes the GRPO training signal. Better environment = better training.', 'green'
)

# ===== PART 4 =====
pdf.add_page()
pdf.chapter_title('Part 4: RLHF - How ChatGPT Was Trained')

pdf.section_title('RLHF = Reinforcement Learning from Human Feedback')
pdf.body_text(
    'This is how ChatGPT, Claude, and other chat models were made helpful and safe.'
)

pdf.code_block(
    'Step 1: Pre-train LLM on internet text (next-token prediction)\n'
    '        -> Can complete text but has no "personality"\n'
    '\n'
    'Step 2: Supervised Fine-Tuning (SFT)\n'
    '        -> Train on (question, ideal answer) pairs\n'
    '        -> Learns FORMAT of good answers\n'
    '\n'
    'Step 3: Train a Reward Model\n'
    '        -> Humans rank answers: A > B > C\n'
    '        -> Model learns to score answers\n'
    '\n'
    'Step 4: RL Fine-Tuning (PPO or GRPO)\n'
    '        -> LLM generates answers\n'
    '        -> Reward model scores them\n'
    '        -> Update LLM to produce higher-scored answers'
)

pdf.section_title('Why YOUR Project is Simpler')
pdf.body_text(
    'Standard RLHF: Human preferences -> Reward Model -> Train LLM\n'
    'YOUR project:  Economic simulation -> Grader Score -> Train LLM\n\n'
    'You do NOT need a reward model because your environment directly provides '
    'a numerical score (0.0 to 1.0). This is simpler and more reliable!'
)

# ===== PART 5 =====
pdf.add_page()
pdf.chapter_title('Part 5: PPO vs GRPO vs DPO')

pdf.section_title('PPO (Proximal Policy Optimization)')
pdf.body_text(
    'The algorithm that trained ChatGPT (Schulman et al., 2017). '
    '"Make good actions more likely, bad actions less likely, but do not change too much." '
    'Needs a critic network (value function). Complex but stable and well-understood.'
)

pdf.section_title('GRPO (Group Relative Policy Optimization) -- YOUR METHOD')
pdf.body_text(
    'Invented by DeepSeek (Jan 2024). Used to train DeepSeek-R1. '
    '"Generate multiple answers, compare them to each other, learn from the best ones." '
    'NO critic network needed -- simpler and cheaper than PPO!'
)

pdf.code_block(
    'How GRPO works:\n'
    '  1. For each prompt, generate N=4 completions\n'
    '  2. Score ALL of them with the environment grader\n'
    '     e.g., rewards = [0.35, 0.72, 0.51, 0.68]\n'
    '  3. Compute RELATIVE advantage:\n'
    '     mean = 0.565\n'
    '     advantages = [-1.4, +1.0, -0.35, +0.75]\n'
    '  4. Update model:\n'
    '     completion with 0.72 -> make MORE likely\n'
    '     completion with 0.35 -> make LESS likely\n'
    '  5. No separate critic network needed!'
)

pdf.important_box(
    'JUDGE ANSWER: "We use GRPO from DeepSeek-R1. For each scenario, the LLM generates '
    '4 different policy responses. Our environment scores each one. The model learns to '
    'generate more high-scoring responses. No reward model needed -- the environment IS '
    'the reward signal."', 'blue'
)

pdf.section_title('DPO (Direct Preference Optimization)')
pdf.body_text(
    'Alternative to PPO/GRPO. Given A is better than B, directly train model to prefer A. '
    'Needs paired preferences. NOT used in your project because GRPO works better with '
    'environment scores. Used by Llama, Zephyr.'
)

pdf.section_title('Algorithm Comparison')
pdf.code_block(
    'Feature          PPO           GRPO (YOURS)   DPO\n'
    '-------          ---           -----------    ---\n'
    'Critic needed    Yes           NO             No\n'
    'Reward model     Yes           NO (uses env)  Yes\n'
    'Compute cost     High          Medium         Low\n'
    'Used by          ChatGPT       DeepSeek-R1    Llama\n'
    'Best for         General RLHF  Env-based RL   Preferences'
)

# ===== PART 6 =====
pdf.add_page()
pdf.chapter_title('Part 6: Unsloth, LoRA & Efficient Training')

pdf.section_title('The Problem: Models Are Too Big')
pdf.body_text(
    'A 1.5B parameter model has 1,500,000,000 numbers to update. Fine-tuning ALL of them '
    'needs ~12GB GPU memory just for weights plus much more for gradients. T4 GPU = 16GB.'
)

pdf.section_title('LoRA (Low-Rank Adaptation) -- Hu et al. 2021')
pdf.body_text(
    'Instead of changing ALL 1.5B parameters, add small "adapter" matrices and only '
    'train THOSE. Original weight: 4096x4096 = 16.7M params. '
    'LoRA adds: A(4096x16) x B(16x4096) = 131K params. '
    'You train ~2M params total = 0.13% of the model!'
)

pdf.code_block(
    'Output = W*x + A*B*x\n'
    '         ^      ^\n'
    '      frozen  trained (LoRA adapters)\n'
    '\n'
    'Rank r=16 controls adapter size:\n'
    '  r=4:  Very small, fast, might underfit\n'
    '  r=16: Good balance (YOUR config)\n'
    '  r=64: Large, slower, might overfit'
)

pdf.section_title('QLoRA (Quantized LoRA)')
pdf.body_text(
    'Load the frozen model in 4-bit precision to save even more memory:\n'
    '  Full precision (fp32): 1.5B x 4 bytes = 6.0 GB\n'
    '  Half precision (fp16): 1.5B x 2 bytes = 3.0 GB\n'
    '  4-bit quantization:    1.5B x 0.5 bytes = 0.75 GB  <-- YOU USE THIS\n\n'
    'With QLoRA, the frozen model uses ~750MB. Plenty of room on a 16GB T4!'
)

pdf.section_title('Unsloth')
pdf.body_text(
    'Python library that makes everything 2x faster. Loads models in 4-bit with optimized '
    'kernels, applies LoRA automatically, uses custom CUDA kernels, reduces memory by ~60%. '
    'One line: FastLanguageModel.from_pretrained("unsloth/Qwen2.5-1.5B-Instruct", load_in_4bit=True)'
)

pdf.important_box(
    'JUDGE ANSWER: "We use Unsloth with 4-bit QLoRA. The frozen 1.5B model takes 750MB, '
    'we train only ~2M LoRA parameters (0.13% of the model), and it runs in under an hour '
    'on a free Colab T4 GPU."', 'green'
)

# ===== PART 7 =====
pdf.add_page()
pdf.chapter_title('Part 7: Multi-Agent RL')

pdf.section_title('What is Multi-Agent RL?')
pdf.body_text(
    'Multiple agents making decisions in the same environment, where each agent\'s actions '
    'affect the others. Your setup: LLM policy advisor vs market speculator consortium.'
)

pdf.section_title('Types of Multi-Agent Interactions')
pdf.body_text(
    'Cooperative: Agents work together (NOT you)\n'
    'Competitive: Zero-sum, opposing goals (NOT you)\n'
    'Mixed: Partially aligned goals (THIS IS YOU!)\n\n'
    'Your market speculator WANTS a stable economy (to invest) but EXPLOITS policy '
    'mistakes (speculative attacks). The LLM must build market confidence through '
    'consistent, credible policy.'
)

pdf.section_title('Your Market Consortium (3 Agents)')
pdf.code_block(
    'Institutional Investor (50% of capital)\n'
    '  Aggressiveness: 0.2 (low)\n'
    '  Strategy: Conservative, follows fundamentals\n'
    '  Role: Stabilizing force\n'
    '\n'
    'Hedge Fund (30% of capital)\n'
    '  Aggressiveness: 0.7 (high)\n'
    '  Strategy: Exploits policy inconsistency\n'
    '  Role: Can trigger SPECULATIVE ATTACKS!\n'
    '\n'
    'Momentum Trader (20% of capital)\n'
    '  Aggressiveness: 0.5 (medium)\n'
    '  Strategy: Follows trends\n'
    '  Role: Amplifies booms AND busts'
)

pdf.important_box(
    'SPECULATIVE ATTACKS: When hedge fund detects low confidence (<0.3) + high inflation '
    '(>6%) + large deficit, it attacks (like Soros vs Bank of England 1992). Capital flees, '
    'GDP drops, unrest increases. The LLM must PREVENT attacks through consistent policy.', 'orange'
)

# ===== PART 8 =====
pdf.add_page()
pdf.chapter_title('Part 8: Curriculum Learning')

pdf.section_title('What is Curriculum Learning?')
pdf.body_text(
    'Start easy, gradually increase difficulty as the agent improves. '
    'Like school: Grade 1 = addition, Grade 5 = fractions, Grade 10 = calculus. '
    'Directly addresses Theme #4 (Self-Improvement). '
    'Inspired by OpenAI\'s Automatic Domain Randomization (ADR).'
)

pdf.section_title('Your 5 Difficulty Levels')
pdf.code_block(
    'Level 0: NORMAL      Standard parameters\n'
    'Level 1: CHALLENGING  1.3x shocks, 1.2x trust decay\n'
    'Level 2: HARD         1.6x shocks, initial deficit\n'
    'Level 3: EXPERT+      2x shocks, +2% inflation, -3 steps\n'
    'Level 4: NIGHTMARE    2.5x shocks, +4% inflation, -5 steps\n'
    '\n'
    'Auto-promote when rolling avg > 0.70\n'
    'Auto-demote  when rolling avg < 0.35'
)

pdf.important_box(
    'JUDGE ANSWER: "Our curriculum implements ADR inspired by OpenAI\'s Rubik\'s Cube paper. '
    'When the agent masters one difficulty, the environment automatically escalates. '
    'This creates recursive skill amplification -- Theme #4 Self-Improvement."', 'blue'
)

# ===== PART 9 =====
pdf.add_page()
pdf.chapter_title('Part 9: Your Reward Function Deep-Dive')

pdf.section_title('Why Reward Design Matters')
pdf.body_text(
    'The reward function is the MOST IMPORTANT part of any RL system. '
    'Bad reward = agent learns the wrong thing (reward hacking).'
)
pdf.code_block(
    'BAD reward:    reward = GDP_growth\n'
    'Agent learns:  Print infinite money (QE=500 every step)\n'
    'Result:        Hyperinflation destroys everything!\n'
    '\n'
    'YOUR reward:   reward = GDP + equality + satisfaction\n'
    '                      - unrest - deficit - volatility\n'
    'Agent learns:  Balanced, consistent economic management'
)

pdf.section_title('Your Reward Formula')
pdf.code_block(
    'reward = (\n'
    '    0.35 * gdp_norm            # 35% - Did economy grow?\n'
    '  + 0.25 * (1.0 - gini)        # 25% - Is inequality low?\n'
    '  + 0.15 * satisfaction         # 15% - Are citizens happy?\n'
    '  - 0.12 * unrest               # 12% - Are citizens rioting?\n'
    '  - 0.08 * deficit_penalty      #  8% - Is budget balanced?\n'
    '  - 0.05 * tax_penalty          #  5% - Taxes too high (>50%)?\n'
    '  - 0.08 * volatility_penalty   #  8% - Did agent flip-flop?\n'
    ')'
)

pdf.section_title('Step Reward vs Final Grader')
pdf.body_text(
    'Per-Step Reward: Given every step, guides learning (DENSE signal). '
    'Helps the agent learn faster during training.\n\n'
    'Final Grader Score: Given at end, evaluates entire trajectory (SPARSE signal). '
    'Checks phase-aware strategy, simultaneous gates, volatility. '
    'This is what matters for your hackathon score.'
)

# ===== PART 10 =====
pdf.add_page()
pdf.chapter_title('Part 10: Judge Q&A Cheat Sheet')
pdf.body_text('ALL three team members should memorize these answers:')

qas = [
    ('Q: "How does the RL training work?"',
     'A: "We use GRPO from DeepSeek-R1. For each economic scenario, the LLM generates 4 '
     'different policy responses. Our environment scores each one using the grader. The '
     'model learns to generate more high-scoring responses. No reward model needed -- the '
     'environment IS the reward signal."'),

    ('Q: "What is the reward function?"',
     'A: "Multi-objective: 35% GDP growth, 25% equality, 15% satisfaction, minus penalties '
     'for unrest, deficit, and policy volatility. The volatility penalty forces consistent '
     'strategies instead of flip-flopping -- just like real economic advisory."'),

    ('Q: "How do you prevent reward hacking?"',
     'A: "Three mechanisms: (1) volatility penalty punishes erratic changes, (2) phase-aware '
     'grader checks entire trajectory not just final state, (3) multi-seed evaluation across '
     '5 seeds ensures generalization."'),

    ('Q: "What makes this different from simple envs?"',
     'A: "28 continuous observations, 8 simultaneous levers (high-dimensional), trust dynamics '
     '(citizens remember past policies), and emergent behaviors like tax evasion, capital '
     'flight, and strikes emerge from citizen-level economics."'),

    ('Q: "Is this really multi-agent?"',
     'A: "Yes -- 3 independent market agents with different risk profiles. The hedge fund can '
     'launch speculative attacks when policy credibility drops. The LLM must maintain market '
     'confidence through consistent policy."'),

    ('Q: "What model and why?"',
     'A: "Qwen 2.5 1.5B with Unsloth 4-bit QLoRA. Small enough for free Colab T4 in under '
     'an hour. After GRPO training, it outperforms GPT-4o-mini zero-shot at 0.788 vs 0.744. '
     'Shows genuine economic reasoning was learned."'),

    ('Q: "How do you measure improvement?"',
     'A: "Three things: (1) reward curves from 0.30 to 0.79, (2) before/after behavior '
     'comparison showing panic vs clean multi-phase strategy, (3) multi-model benchmark '
     'where our trained 1.5B beats GPT-4o-mini on expert tasks."'),
]

for q, a in qas:
    pdf.sub_section(q)
    pdf.body_text(a)

# ===== PART 12: QUICK REFERENCE =====
pdf.add_page()
pdf.chapter_title('Part 12: Quick Reference Card (PRINT THIS)')

pdf.code_block(
    '  AGENT:      LLM (Qwen 1.5B, 4-bit QLoRA via Unsloth)\n'
    '  ENV:        SocialContract-v0 (100 citizens, 5 tasks)\n'
    '  ACTIONS:    8 continuous policy levers\n'
    '  OBS:        28 economic indicators (partially obs.)\n'
    '  REWARD:     Multi-objective [0,1] per step\n'
    '  GRADER:     Phase-aware, trajectory-based [0,1]\n'
    '  TRAINING:   GRPO via TRL + Unsloth on Colab T4\n'
    '  MULTI-AGT:  3-agent market consortium\n'
    '  CURRICULUM: 5 difficulty levels, auto-scaling\n'
    '\n'
    '  SCORES:\n'
    '    Random baseline:  0.290\n'
    '    Smart heuristic:  0.656\n'
    '    GPT-4o-mini:      0.744\n'
    '    GRPO fine-tuned:  0.788\n'
    '\n'
    '  KEY PAPERS:\n'
    '    GRPO: DeepSeek-R1 (2024)\n'
    '    LoRA: Hu et al. (2021)\n'
    '    PPO:  Schulman et al. (2017)\n'
    '    RLHF: Ouyang et al. (2022)'
)

pdf.ln(5)
pdf.section_title('5 Things ALL Members Must Be Able to Say')
pdf.body_text(
    '1. "We use GRPO" -- not PPO, not DPO. GRPO from DeepSeek-R1.\n\n'
    '2. "The environment IS the reward signal" -- no separate reward model.\n\n'
    '3. "8 continuous action levers, 28 observation fields" -- shows complexity.\n\n'
    '4. "Our trained 1.5B model beats GPT-4o-mini zero-shot" -- the punchline.\n\n'
    '5. "Phase-aware grading checks the full trajectory" -- not just final state.'
)

pdf.important_box(
    'GOOD LUCK! You have a genuinely strong project. The environment is novel, the '
    'reward function is well-designed, the training pipeline is complete, and the '
    'multi-agent + curriculum features address multiple hackathon themes. '
    'Now go practice the pitch and nail the delivery.', 'green'
)

# Save
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RL_Crash_Course.pdf')
pdf.output(out_path)
print(f"PDF saved to: {out_path}")
print(f"Total pages: {pdf.page_no()}")
