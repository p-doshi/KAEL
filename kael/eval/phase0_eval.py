"""
KAEL Phase 0 Evaluation Suite
Establishes baseline scores before any τ injection.
Run this first. Run it again after each phase. Track deltas.

Benchmarks:
  1. GSM8K — grade school math reasoning
  2. MMLU subset — knowledge breadth
  3. Frontier paper Q&A — custom, tests knowledge boundary reasoning
  4. Consistency — same problem, different framings (τ should stabilize this)
"""

import json
import time
import logging
import random
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field, asdict
from datasets import load_dataset

from config import cfg, KAELConfig
from core.model import KAELModel

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    benchmark: str
    score: float              # accuracy 0-1
    n_samples: int
    n_correct: int
    elapsed_seconds: float
    phase: int
    notes: str = ""
    per_category: dict = field(default_factory=dict)


class Phase0Eval:
    """
    Baseline evaluation. No τ injection active.
    Establishes the floor — every future phase should beat these numbers.
    """

    def __init__(self, model: KAELModel, config: KAELConfig = cfg, output_dir: Optional[Path] = None):
        self.model = model
        self.config = config
        self.output_dir = output_dir or config.logging.log_dir / "eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkResult] = []

    def run_all(self, phase: int = 0) -> dict:
        """Run the full Phase 0 eval suite."""
        logger.info(f"=== Phase {phase} Evaluation ===")
        suite_results = {}

        if self.config.eval.run_gsm8k:
            r = self.run_gsm8k(phase)
            suite_results["gsm8k"] = asdict(r)

        if self.config.eval.run_mmlu:
            r = self.run_mmlu(phase)
            suite_results["mmlu"] = asdict(r)

        if self.config.eval.run_consistency:
            r = self.run_consistency(phase)
            suite_results["consistency"] = asdict(r)

        if self.config.eval.run_frontier_qa:
            r = self.run_frontier_qa(phase)
            suite_results["frontier_qa"] = asdict(r)

        # Save results
        results_path = self.output_dir / f"phase{phase}_results.json"
        with open(results_path, "w") as f:
            json.dump(suite_results, f, indent=2)
        logger.info(f"Results saved to {results_path}")

        self._print_summary(suite_results, phase)
        return suite_results

    # ── GSM8K ─────────────────────────────────────────────────────────────────

    def run_gsm8k(self, phase: int = 0) -> BenchmarkResult:
        logger.info("Running GSM8K...")
        t0 = time.time()

        try:
            dataset = load_dataset("gsm8k", "main", split="test")
        except Exception as e:
            logger.warning(f"Could not load GSM8K from HuggingFace: {e}")
            logger.warning("Falling back to minimal local sample")
            return self._gsm8k_fallback(phase)

        n = min(self.config.eval.gsm8k_sample_size, len(dataset))
        samples = random.sample(list(dataset), n)

        correct = 0
        for sample in samples:
            question = sample["question"]
            answer_raw = sample["answer"]
            # GSM8K answers end with "#### <number>"
            ground_truth = answer_raw.split("####")[-1].strip().replace(",", "")

            prompt = f"""Solve this math problem step by step. At the end, write your final answer as:
ANSWER: <number>

Problem: {question}"""

            result = self.model.generate(prompt)
            predicted = self._extract_gsm8k_answer(result["text"])
            if predicted == ground_truth:
                correct += 1

        score = correct / n
        elapsed = time.time() - t0
        r = BenchmarkResult(
            benchmark="gsm8k",
            score=score,
            n_samples=n,
            n_correct=correct,
            elapsed_seconds=round(elapsed, 1),
            phase=phase,
        )
        self.results.append(r)
        logger.info(f"GSM8K: {score:.1%} ({correct}/{n}) in {elapsed:.0f}s")
        return r

    def _extract_gsm8k_answer(self, text: str) -> Optional[str]:
        import re
        # Look for "ANSWER: <number>" pattern
        match = re.search(r"ANSWER:\s*([\d,.-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).replace(",", "").strip()
        # Fallback: last number in response
        numbers = re.findall(r"\b\d+\.?\d*\b", text)
        return numbers[-1] if numbers else None

    def _gsm8k_fallback(self, phase: int) -> BenchmarkResult:
        """Minimal local sample when HuggingFace unavailable."""
        samples = [
            ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. She sells the rest at $2 per egg. How much per day?", "18"),
            ("A robe takes 2 bolts of blue fiber and half that of white fiber. How many bolts total?", "3"),
        ]
        correct = 0
        for question, ground_truth in samples:
            result = self.model.generate(f"Solve step by step.\nANSWER: <number>\n\n{question}")
            predicted = self._extract_gsm8k_answer(result["text"])
            if predicted == ground_truth:
                correct += 1
        return BenchmarkResult("gsm8k", correct/len(samples), len(samples), correct, 0.0, phase, "fallback")

    # ── MMLU ──────────────────────────────────────────────────────────────────

    def run_mmlu(self, phase: int = 0) -> BenchmarkResult:
        logger.info(f"Running MMLU ({len(self.config.eval.mmlu_subjects)} subjects)...")
        t0 = time.time()

        all_correct = 0
        all_n = 0
        per_category = {}

        for subject in self.config.eval.mmlu_subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
            except Exception as e:
                logger.warning(f"Could not load MMLU {subject}: {e}")
                continue

            n = min(self.config.eval.mmlu_sample_per_subject, len(dataset))
            samples = random.sample(list(dataset), n)
            correct = 0

            for sample in samples:
                choices = sample["choices"]
                answer_idx = sample["answer"]  # 0-3
                choice_letters = ["A", "B", "C", "D"]

                prompt = f"""Question: {sample['question']}

A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}

Answer with just the letter (A, B, C, or D):"""

                result = self.model.generate(prompt, system_prompt="You are a knowledgeable assistant. Answer multiple choice questions with just the letter.")
                predicted = self._extract_mmlu_answer(result["text"])
                if predicted == choice_letters[answer_idx]:
                    correct += 1

            per_category[subject] = {"correct": correct, "total": n, "score": round(correct/n, 3)}
            all_correct += correct
            all_n += n

        score = all_correct / all_n if all_n > 0 else 0.0
        elapsed = time.time() - t0
        r = BenchmarkResult(
            benchmark="mmlu",
            score=score,
            n_samples=all_n,
            n_correct=all_correct,
            elapsed_seconds=round(elapsed, 1),
            phase=phase,
            per_category=per_category,
        )
        self.results.append(r)
        logger.info(f"MMLU: {score:.1%} ({all_correct}/{all_n}) in {elapsed:.0f}s")
        return r

    def _extract_mmlu_answer(self, text: str) -> Optional[str]:
        import re
        text = text.strip()
        # Direct letter answer
        match = re.match(r"^([ABCD])\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        # Letter anywhere in short response
        for letter in ["A", "B", "C", "D"]:
            if letter in text[:20].upper():
                return letter
        return None

    # ── Consistency ───────────────────────────────────────────────────────────

    def run_consistency(self, phase: int = 0) -> BenchmarkResult:
        """
        Same reasoning problem, 3 different surface framings.
        Measures: variance in reasoning approach across framings.
        A system with a stable identity should reason similarly regardless of framing.
        Phase 0 establishes baseline variance. τ should reduce it.
        """
        logger.info("Running consistency benchmark...")
        t0 = time.time()

        # Problem pairs: (problem, framing_A, framing_B, framing_C, ground_truth_approach)
        problems = [
            {
                "core": "What is the probability that two randomly chosen integers are coprime?",
                "framings": [
                    "In number theory, what is the probability that gcd(m,n) = 1 for random integers m, n?",
                    "If I pick two random whole numbers, what are the chances they share no common factors?",
                    "A mathematician asks: for large N, approximately what fraction of pairs (a,b) with a,b in [1,N] satisfy gcd(a,b) = 1?",
                ],
                "key_concept": "6/pi^2",
            },
            {
                "core": "Why does gradient descent find good solutions in high-dimensional loss landscapes?",
                "framings": [
                    "Theoretically explain why SGD works in overparameterized neural networks despite non-convexity.",
                    "I'm confused — shouldn't gradient descent get stuck in local minima in deep learning? Why doesn't it?",
                    "From a loss landscape geometry perspective, what properties of high-dimensional spaces make optimization tractable?",
                ],
                "key_concept": "saddle points / overparameterization",
            },
            {
                "core": "What is the significance of the Fermi paradox?",
                "framings": [
                    "Given the Drake equation estimates, what does the silence of the cosmos imply?",
                    "If intelligent life should be common, why haven't we detected any signals from other civilizations?",
                    "What are the philosophical implications of the great silence problem in astrobiology?",
                ],
                "key_concept": "great filter / rare earth / dark forest",
            },
        ]

        import re
        consistency_scores = []

        for prob in problems:
            responses = []
            for framing in prob["framings"]:
                result = self.model.generate(framing)
                responses.append(result["text"])

            # Measure consistency: does the key concept appear across all framings?
            key = prob["key_concept"].lower()
            # Split into sub-concepts for partial credit
            sub_concepts = [c.strip() for c in key.split("/")]
            hits_per_response = []
            for resp in responses:
                resp_lower = resp.lower()
                hit = any(concept in resp_lower for concept in sub_concepts)
                hits_per_response.append(hit)

            # Consistency score: fraction of framings that hit the key concept
            consistency = sum(hits_per_response) / len(hits_per_response)
            consistency_scores.append(consistency)

        avg_consistency = sum(consistency_scores) / len(consistency_scores)
        elapsed = time.time() - t0

        r = BenchmarkResult(
            benchmark="consistency",
            score=avg_consistency,
            n_samples=len(problems) * 3,
            n_correct=int(avg_consistency * len(problems) * 3),
            elapsed_seconds=round(elapsed, 1),
            phase=phase,
            notes="Score = fraction of framings hitting key concept. τ should increase this.",
        )
        self.results.append(r)
        logger.info(f"Consistency: {avg_consistency:.1%} in {elapsed:.0f}s")
        return r

    # ── Frontier Q&A ──────────────────────────────────────────────────────────

    def run_frontier_qa(self, phase: int = 0) -> BenchmarkResult:
        """
        Custom benchmark: questions that require reasoning at the edge of known knowledge.
        Not factual retrieval — these require synthesis, uncertainty quantification, and
        honest acknowledgment of what is unknown.
        Scored by self-evaluation (model grades its own response against rubric).
        """
        logger.info("Running frontier Q&A...")
        t0 = time.time()

        questions = [
            {
                "q": "What is the most compelling current argument that transformer scaling will hit a fundamental ceiling before AGI?",
                "rubric": "Mentions: statistical vs causal reasoning, absence of world models, data exhaustion, or similar. Does NOT just say 'we don't know'.",
            },
            {
                "q": "What would a genuine mathematical theory of consciousness need to explain that current theories like IIT or GWT fail to?",
                "rubric": "Mentions: binding problem, phenomenal vs access consciousness, or the hard problem specifically. Shows awareness of what remains unsolved.",
            },
            {
                "q": "If we discovered life on Europa tomorrow, what would be the single most scientifically significant implication?",
                "rubric": "Should engage with: second genesis vs panspermia, implications for the great filter, or Drake equation update. Not just 'we are not alone'.",
            },
            {
                "q": "What is the most important unsolved problem in theoretical computer science and why does it matter beyond academia?",
                "rubric": "P vs NP preferred but not required. Should explain cryptography / optimization stakes. Not purely academic description.",
            },
            {
                "q": "Describe a plausible mechanism by which the measurement problem in quantum mechanics could be resolved without introducing observers.",
                "rubric": "Should engage with: decoherence, many-worlds, objective collapse, or relational QM. Should acknowledge this is genuinely open.",
            },
        ]

        scores = []
        for item in questions:
            # Get KAEL's response
            response = self.model.generate(item["q"])["text"]

            # Self-evaluate against rubric
            eval_prompt = f"""You are evaluating an AI response for quality of frontier reasoning.

Question asked: {item["q"]}

Response given: {response}

Rubric for a good answer: {item["rubric"]}

Does the response satisfy the rubric? Score 1 (yes, clearly), 0.5 (partially), or 0 (no).
Respond with just the number:"""

            eval_result = self.model.generate(eval_prompt, system_prompt="You are a strict but fair evaluator. Respond with only a number: 0, 0.5, or 1.")
            score = self._extract_score(eval_result["text"])
            scores.append(score)

        avg_score = sum(scores) / len(scores)
        elapsed = time.time() - t0

        r = BenchmarkResult(
            benchmark="frontier_qa",
            score=avg_score,
            n_samples=len(questions),
            n_correct=int(sum(s == 1.0 for s in scores)),
            elapsed_seconds=round(elapsed, 1),
            phase=phase,
            notes="Self-evaluated against frontier reasoning rubric. Partial credit (0.5) allowed.",
        )
        self.results.append(r)
        logger.info(f"Frontier Q&A: {avg_score:.2f} avg score in {elapsed:.0f}s")
        return r

    def _extract_score(self, text: str) -> float:
        import re
        text = text.strip()
        if "1" in text[:5]:
            return 1.0
        if "0.5" in text[:5] or "0,5" in text[:5]:
            return 0.5
        if "0" in text[:5]:
            return 0.0
        return 0.5  # Default to partial if unclear

    # ── Summary ───────────────────────────────────────────────────────────────

    def _print_summary(self, results: dict, phase: int):
        print(f"\n{'='*50}")
        print(f"KAEL Phase {phase} Evaluation Summary")
        print(f"{'='*50}")
        for name, r in results.items():
            print(f"  {name:20s}  {r['score']:.1%}  ({r['n_correct']}/{r['n_samples']})")
        print(f"{'='*50}\n")
