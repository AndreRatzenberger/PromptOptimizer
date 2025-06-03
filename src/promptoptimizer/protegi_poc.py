
"""
Paper: "Automatic Prompt Optimization with “Gradient Descent” and Beam Search" Reid Pryzant
"""

###############################################################################
# Standard library
###############################################################################
import argparse
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

###############################################################################
# Third‑party deps
###############################################################################
import litellm                 # ⇦ unified chat wrapper

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

###############################################################################
# ----- Metrics helpers ------------------------------------------------------ #
###############################################################################
_LEVEL_METRIC_FN = {
    "f1": f1_score,
    "accuracy": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
}

###############################################################################
# ----- LiteLLM thin wrapper ------------------------------------------------- #
###############################################################################
def chat_complete(
    messages: List[Dict],
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 8192,
) -> str:
    """Single synchronous completion with exponential back-off.

    Removes any <think>...</think> blocks (tags plus enclosed text)
    from the model’s response before returning it.
    """
    backoff = 1.0
    while True:
        try:
            resp = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            raw_content = resp["choices"][0]["message"]["content"]

            # Strip entire <think>...</think> sections (tags + contents, multi-line safe)
            cleaned_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.S)

            return cleaned_content.strip()

        except Exception as e:  # noqa: BLE001
            print(f"[litellm] {e} – retrying in {backoff:.1f}s")
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

###############################################################################
# ----- Data loading & evaluation ------------------------------------------- #
###############################################################################

def _hf_sst2() -> pd.DataFrame:
    """Load GLUE/SST‑2 and return (text,label) DataFrame."""
    ds = load_dataset("glue", "sst2", split="train")
    df = ds.to_pandas()[["sentence", "label"]]
    df = df.rename(columns={"sentence": "text"})
    return df.astype({"label": int})


def load_dataset_any(path_or_name: str | None) -> pd.DataFrame:
    if path_or_name is None or path_or_name.lower() == "sst2":
        return _hf_sst2()

    if os.path.isfile(path_or_name):
        df = pd.read_csv(path_or_name)
        if "sentence" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"sentence": "text"})
    else:
        ds = load_dataset(path_or_name)
        split = ds["train"] if "train" in ds else list(ds.values())[0]
        df = split.to_pandas()
        if "sentence" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"sentence": "text"})

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    return df[["text", "label"]].dropna().astype({"label": int})


def parse_yes_no(output: str) -> int:
    m = re.search(r"yes|no", output, re.I)
    if not m:
        return 1  # default to positive
    return 1 if m.group(0).lower() == "yes" else 0


def evaluate_prompt(prompt: str, batch: Sequence[Tuple[str, int]], *, model: str, temperature: float) -> Tuple[List[int], List[int]]:
    preds: List[int] = []
    for text, _ in tqdm(batch, desc="Evaluating prompt", leave=False):
        full_prompt = f"{prompt}\nText: {text}\nLabel:"
        out = chat_complete([
            {"role": "user", "content": full_prompt}
        ], model=model, temperature=temperature, max_tokens=1024)
        preds.append(parse_yes_no(out))
    return preds, preds  # second value kept for compatibility

###############################################################################
# ----- ProTeGi algorithm ---------------------------------------------------- #
###############################################################################
_GRADIENT_TMPL = (
    "I'm trying to write a zero‑shot classifier prompt.\n"
    "My current prompt is:\n\"{prompt}\"\n"
    "But this prompt gets the following examples wrong:\n{error_string}\n"
    "Give {num_feedbacks} reasons why the prompt could have gotten these examples wrong.\n"
    "Wrap each reason with <START> and <END>."
)

_EDIT_TMPL = (
    "I'm trying to write a zero‑shot classifier.\n"
    "My current prompt is:\n\"{prompt}\"\n"
    "But it gets the following examples wrong:\n{error_str}\n"
    "Based on these examples the problem with this prompt is that {gradient}.\n"
    "Based on the above information, I wrote {steps_per_gradient} different improved prompts, each wrapped with <START> and <END>.\n"
    "The prompts are:"
)

_PARAPHRASE_TMPL = (
    "Generate a variation of the following instruction while keeping the meaning.\n"
    "Input: {prompt_instruction}\nOutput:"
)

###############################################################################
@dataclass
class ProTeGiConfig:
    beam_width: int = 4
    search_depth: int = 6
    minibatch_size: int = 32
    gradients_per_group: int = 2
    steps_per_gradient: int = 1
    mc_variations: int = 2
    eval_model: str = "ollama_chat/qwen3"
    eval_temperature: float = 0.0
    gen_model: str = "ollama_chat/qwen3"
    gen_temperature: float = 1.0
    metric: str = "f1"  # must be key of _LEVEL_METRIC_FN
    max_prompt_length: int = 2048  # max length for the prompt, used to truncate if needed


@dataclass
class ProTeGi:
    config: ProTeGiConfig
    train_data: List[Tuple[str, int]]

    # ----------------------------------------------------------
    def _metric(self, y_true: List[int], y_pred: List[int]) -> float:
        fn = _LEVEL_METRIC_FN[self.config.metric]
        return fn(y_true, y_pred, zero_division=0)

    # ----------------------------------------------------------
    def _format_errors(self, errs: List[Tuple[str, int, int]]) -> str:
        lines = [f"Text: {x}\nLabel: {y} Prediction: {p}" for x, y, p in errs]
        return "\n".join(lines)

    # Gradient generation
    def _generate_gradients(self, prompt: str, errs: List[Tuple[str, int, int]]) -> List[str]:
        msg = _GRADIENT_TMPL.format(
            prompt=prompt,
            error_string=self._format_errors(errs),
            num_feedbacks=self.config.gradients_per_group,
        )
        raw = chat_complete([
            {"role": "user", "content": msg}
        ], model=self.config.gen_model, temperature=self.config.gen_temperature)
        return re.findall(r"<START>(.*?)<END>", raw, re.S)

    # Prompt editing
    def _apply_gradient(self, prompt: str, errs: List[Tuple[str, int, int]], gradient: str) -> List[str]:
        msg = _EDIT_TMPL.format(
            prompt=prompt,
            error_str=self._format_errors(errs),
            gradient=gradient,
            steps_per_gradient=self.config.steps_per_gradient,
        )
        raw = chat_complete([
            {"role": "user", "content": msg}
        ], model=self.config.gen_model, temperature=self.config.gen_temperature)
        return re.findall(r"<START>(.*?)<END>", raw, re.S)

    # Simple paraphrasing for exploration
    def _paraphrase(self, prompt: str) -> List[str]:
        raw = chat_complete([
            {"role": "user", "content": _PARAPHRASE_TMPL.format(prompt_instruction=prompt)}
        ], model=self.config.gen_model, temperature=self.config.gen_temperature)
        variants = re.split(r"\n\d+\.\s", raw)
        variants = [v.strip() for v in variants if v.strip() and len(v) > 20]
        return variants[: self.config.mc_variations]

    # ----------------------------------------------------------      
    def _expand(self, prompt: str) -> List[str]:
        batch = random.sample(self.train_data, min(self.config.minibatch_size, len(self.train_data)))
        preds, _ = evaluate_prompt(prompt, batch, model=self.config.eval_model, temperature=self.config.eval_temperature)
        errs = [(x, y, p) for (x, y), p in zip(batch, preds) if y != p]
        
        accuracy = (len(batch) - len(errs)) / len(batch)
        print(f"      Batch accuracy: {accuracy:.2%} ({len(errs)}/{len(batch)} errors)")
        
        if not errs:
            print("      ✅ Perfect accuracy - no expansion needed")
            return []
        
        print(f"      🔍 Found {len(errs)} errors to analyze")
        
        # Show a few example errors
        if len(errs) > 0:
            print("      Example errors:")
            for i, (text, gold, pred) in enumerate(errs[:2]):  # Show first 2 errors
                sentiment = "positive" if gold == 1 else "negative"
                pred_sentiment = "positive" if pred == 1 else "negative"
                print(f"        {i+1}. \"{text}\" → Expected: {sentiment}, Got: {pred_sentiment}")  # FULL TEXT
        
        groups = [errs[i:i + 4] for i in range(0, len(errs), 4)]
        successors: List[str] = []
        
        print(f"      🧠 Processing {len(groups)} error groups...")
        
        for g_idx, g in enumerate(tqdm(groups, desc="Processing error groups", leave=False)):
            gradients = self._generate_gradients(prompt, g)
            print(f"        Group {g_idx+1}: Generated {len(gradients)} gradient insights")
            
            for gr_idx, gr in enumerate(tqdm(gradients, desc="Applying gradients", leave=False)):
                edits = self._apply_gradient(prompt, g, gr)
                print(f"          Gradient {gr_idx+1}: \"{gr}\" → {len(edits)} edits")  # FULL GRADIENT TEXT
                
                for e in edits:
                    successors.append(e)
                    paraphrases = self._paraphrase(e)
                    successors.extend(paraphrases)
        
        unique_successors = list(dict.fromkeys([s.strip() for s in successors if s.strip()]))
        print(f"      ✨ Generated {len(successors)} total variants → {len(unique_successors)} unique")
        
        return unique_successors

    # ----------------------------------------------------------
    def _select_ucb(self, cand: List[str], k: int, pulls_per_iter: int = 30) -> List[str]:
        if not cand:
            return []
            
        print(f"      🎯 Evaluating {len(cand)} candidates using UCB bandit algorithm...")
        print(f"      📊 Performing {pulls_per_iter} evaluations to estimate performance")
        
        score_est: Dict[int, float] = defaultdict(float)
        pulls: Dict[int, int] = defaultdict(int)
        
        for t in tqdm(range(1, pulls_per_iter + 1), desc="UCB bandit selection", leave=False):
            if t <= len(cand):
                idx = t - 1
            else:
                ucb = [score_est[i] + 2 * math.sqrt(math.log(t) / max(1, pulls[i])) for i in range(len(cand))]
                idx = max(range(len(cand)), key=lambda i: ucb[i])
            x, y = random.choice(self.train_data)
            pred, _ = evaluate_prompt(cand[idx], [(x, y)], model=self.config.eval_model, temperature=self.config.eval_temperature)
            reward = 1.0 if pred[0] == y else 0.0
            pulls[idx] += 1
            score_est[idx] += (reward - score_est[idx]) / pulls[idx]
        
        top = sorted(range(len(cand)), key=lambda i: score_est[i], reverse=True)[:k]
        
        # Show selection results
        print(f"      🏆 Selected top {k} prompts based on estimated performance:")        
        for i, idx in enumerate(top[:3]):  # Show top 3
            score = score_est[idx]
            evaluations = pulls[idx]
            print(f"        {i+1}. Score: {score:.3f} ({evaluations} evals)")
            print_full_prompt(cand[idx], f"Selected Prompt #{i+1}")  # FULL PROMPT
        
        return [cand[i] for i in top]
    
    # ----------------------------------------------------------
    def optimise(self, init_prompt: str) -> str:
        print_section("PROMPT OPTIMIZATION STARTED")
        print_prompt_box(init_prompt, "Initial Prompt")  # Always show full prompt
        
        # Evaluate initial prompt
        print_subsection("Initial Prompt Evaluation")
        gold = [y for _, y in self.train_data]
        init_preds, _ = evaluate_prompt(init_prompt, self.train_data, model=self.config.eval_model, temperature=self.config.eval_temperature)
        init_score = self._metric(gold, init_preds)
        print(f"📊 Initial prompt {self.config.metric}: {init_score:.3f}")
        
        beam = [init_prompt]
        all_scores_history = []
        
        for step in tqdm(range(self.config.search_depth), desc="Optimization steps"):
            print_section(f"OPTIMIZATION STEP {step + 1}/{self.config.search_depth}")
            print(f"🔍 Current beam size: {len(beam)} prompts")            # Show current beam prompts
            if step > 0:  # Skip for initial step since we already showed it
                print_subsection("Current Best Prompts in Beam")
                # ALWAYS show ALL prompts in beam - FULL LENGTH
                for i, p in enumerate(beam):
                    print_full_prompt(p, f"Beam Prompt #{i+1}")
            
            all_succ: List[str] = []
            expansion_stats = {"total_successors": 0, "unique_successors": 0}
            
            print_subsection("Prompt Expansion Phase")
            for i, p in enumerate(tqdm(beam, desc="Expanding prompts")):
                print(f"\n🔧 Expanding prompt #{i+1}...")
                successors = self._expand(p)
                expansion_stats["total_successors"] += len(successors)
                all_succ.extend(successors)
                print(f"   Generated {len(successors)} variants")
            
            if not all_succ:
                print("\n⚠️  No successors generated - optimization complete!")
                break
              # Remove duplicates and show stats
            unique_succ = list(dict.fromkeys(all_succ))
            expansion_stats["unique_successors"] = len(unique_succ)
            
            print("\n📈 Expansion Results:")
            print(f"   • Total generated: {expansion_stats['total_successors']}")
            print(f"   • Unique variants: {expansion_stats['unique_successors']}")
            print(f"   • Duplicates removed: {expansion_stats['total_successors'] - expansion_stats['unique_successors']}")
              # Show a few example generated prompts
            if len(unique_succ) > 0:
                print("\n🔍 Sample generated prompts:")
                for i, prompt in enumerate(unique_succ[:3]):  # Show first 3
                    print(f"   {i+1}. \"{prompt}\"")  # Show FULL prompt, no truncation
                if len(unique_succ) > 3:
                    print(f"   ... and {len(unique_succ) - 3} more variants")
            
            print_subsection("Bandit Selection Phase")
            print(f"🎯 Selecting top {self.config.beam_width} prompts using UCB bandit algorithm...")
            beam = self._select_ucb(unique_succ, self.config.beam_width)
            beam = list(dict.fromkeys(beam))[: self.config.beam_width]
            
            # Evaluate beam prompts and show performance
            beam_scores = []
            for p in beam:
                # Quick evaluation on subset for interim results
                subset = random.sample(self.train_data, min(20, len(self.train_data)))
                preds, _ = evaluate_prompt(p, subset, model=self.config.eval_model, temperature=self.config.eval_temperature)
                subset_gold = [y for _, y in subset]
                score = self._metric(subset_gold, preds)
                beam_scores.append((p, score))
            
            beam_scores.sort(key=lambda x: x[1], reverse=True)
            all_scores_history.append([score for _, score in beam_scores])
            
            print_metrics_table(beam_scores, f"Step {step+1} Beam Performance (subset evaluation)")
            print(f"🏆 Best score this step: {beam_scores[0][1]:.3f}")
            
        print_section("FINAL EVALUATION")
        print("🔍 Evaluating final beam on complete training set...")
        
        best_prompt, best_score = None, -1.0
        final_results = []
        
        for i, p in enumerate(tqdm(beam, desc="Final evaluation")):
            preds, _ = evaluate_prompt(p, self.train_data, model=self.config.eval_model, temperature=self.config.eval_temperature)
            score = self._metric(gold, preds)
            final_results.append((p, score))
            if score > best_score:
                best_prompt, best_score = p, score
        
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        print_metrics_table(final_results, "Final Results - All Beam Prompts")
        
        print_section("OPTIMIZATION SUMMARY")
        print(f"📊 Initial {self.config.metric}: {init_score:.3f}")
        print(f"🏆 Final {self.config.metric}: {best_score:.3f}")
        print(f"📈 Improvement: {best_score - init_score:+.3f} ({((best_score - init_score) / max(init_score, 0.001) * 100):+.1f}%)")
        print(f"🔄 Optimization steps completed: {min(step + 1, self.config.search_depth)}")
        
        return best_prompt

###############################################################################
# ----- Output formatting utilities ------------------------------------------ #
###############################################################################

def print_section(title: str, width: int = 80):
    """Print a formatted section header."""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width)

def print_subsection(title: str, width: int = 80):
    """Print a formatted subsection header."""
    print("\n" + "-" * width)
    print(f" {title} ")
    print("-" * width)

def print_prompt_box(prompt: str, title: str = "Prompt"):
    """Print a prompt in a formatted box - ALWAYS FULL, NEVER TRUNCATED."""
    lines = prompt.split('\n')
    
    # Always show ALL lines - no truncation
    display_lines = lines
    
    # Calculate width based on longest line
    content_width = max(len(line) for line in display_lines + [title]) + 4
    box_width = max(content_width, len(title) + 8)  # No width cap
    
    print(f"\n┌─ {title} " + "─" * (box_width - len(title) - 4) + "┐")
    for line in display_lines:
        # NEVER truncate - if line is too long, expand the box
        if len(line) > box_width - 4:
            box_width = len(line) + 4
            # Reprint header with new width
            print(f"├─ {title} " + "─" * (box_width - len(title) - 4) + "┤")
        print(f"│ {line:<{box_width-4}} │")
    print("└" + "─" * (box_width - 2) + "┘")

def print_full_prompt(prompt: str, title: str = "Full Prompt"):
    """Print a prompt without any truncation."""
    print(f"\n{'='*80}")
    print(f"{title}:")
    print('='*80)
    print(prompt)
    print('='*80)

def print_metrics_table(prompts_scores: List[Tuple[str, float]], title: str = "Prompt Performance"):
    """Print a table of prompts and their scores - ALWAYS SHOW FULL PROMPTS."""
    print(f"\n{title}:")
    print("=" * 120)
    
    # ALWAYS show full prompts - no truncation ever
    for i, (prompt, score) in enumerate(prompts_scores):
        print(f"\n{i+1:2d}. Score: {score:.3f}")
        print_full_prompt(prompt, f"Prompt #{i+1}")
        print("-" * 120)

###############################################################################

# ----- CLI driver ----------------------------------------------------------- #
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="ProTeGi proof‑of‑concept optimiser (litellm edition)")
    parser.add_argument("--dataset", help="CSV path or HF‑datasets ID (default SST‑2)")
    parser.add_argument("--initial_prompt_file", help="File containing starting prompt")
    parser.add_argument("--beam", type=int, default=2, help="Beam width")
    parser.add_argument("--steps", type=int, default=3, help="Search depth")
    parser.add_argument("--metric", choices=list(_LEVEL_METRIC_FN), default="f1")
    parser.add_argument("--gen_model", default="azure/gpt-4.1")
    parser.add_argument("--eval_model", default="ollama_chat/llama3.2:1b")
    parser.add_argument("--temp_eval", type=float, default=0.0)
    parser.add_argument("--temp_gen", type=float, default=1.0)
    args = parser.parse_args()

    print_section("PROTEGI PROMPT OPTIMIZER")
    print("🚀 ProTeGi: Prompt Optimization with Textual Gradients")
    print("📄 Paper: 'Automatic Prompt Optimization with \"Gradient Descent\" and Beam Search'")
    print("👥 Authors: Reid Pryzant et al. (2023)")

    # Dataset ----------------------------------------------------------------
    print_subsection("Dataset Loading")
    print(f"📊 Loading dataset: {args.dataset or 'SST-2 (default)'}")
    df = load_dataset_any(args.dataset)
    train = list(df.itertuples(index=False, name=None))
    
    print(f"📈 Original dataset size: {len(train)} samples")
    
    # get 100 ranom samples
    random.shuffle(train)
    train = train[:100]  # limit to 100 samples for quick testing
    
    print(f"🎯 Using subset for optimization: {len(train)} samples")
    
    # Show dataset statistics
    pos_count = sum(1 for _, label in train if label == 1)
    neg_count = len(train) - pos_count
    print(f"   • Positive samples: {pos_count} ({pos_count/len(train):.1%})")
    print(f"   • Negative samples: {neg_count} ({neg_count/len(train):.1%})")

    # Initial prompt ---------------------------------------------------------
    print_subsection("Initial Prompt Setup")
    if args.initial_prompt_file:
        print(f"📂 Loading prompt from file: {args.initial_prompt_file}")
        with open(args.initial_prompt_file, "r", encoding="utf-8") as f:
            init_prompt = f.read().strip()
    else:
        print("📝 Using default sentiment classification prompt")
        init_prompt = (
            "You are a sentiment classifier. Given a piece of text, decide whether the sentiment is positive. "
            "Answer with 'Yes' if positive and 'No' if negative."
        )

    # Configuration ----------------------------------------------------------
    print_subsection("Optimization Configuration")
    cfg = ProTeGiConfig(
        beam_width=args.beam,
        search_depth=args.steps,
        eval_model=args.eval_model,
        gen_model=args.gen_model,
        eval_temperature=args.temp_eval,
        gen_temperature=args.temp_gen,
        metric=args.metric,
    )
    
    print("🔧 Configuration:")
    print(f"   • Beam width: {cfg.beam_width}")
    print(f"   • Search depth: {cfg.search_depth} steps")
    print(f"   • Evaluation model: {cfg.eval_model}")
    print(f"   • Generation model: {cfg.gen_model}")
    print(f"   • Evaluation temperature: {cfg.eval_temperature}")
    print(f"   • Generation temperature: {cfg.gen_temperature}")
    print(f"   • Optimization metric: {cfg.metric}")
    print(f"   • Minibatch size: {cfg.minibatch_size}")
    print(f"   • Gradients per group: {cfg.gradients_per_group}")

    # Run optimization -------------------------------------------------------
    optimiser = ProTeGi(cfg, train)
    best = optimiser.optimise(init_prompt)

    print_section("FINAL OPTIMIZED PROMPT")
    print_prompt_box(best, "🏆 Best Prompt Found")  # Always show full final prompt
    
    print("\n🎉 Optimization completed successfully!")
    print("💡 You can now use this optimized prompt for your sentiment classification tasks.")


if __name__ == "__main__":
    main()
