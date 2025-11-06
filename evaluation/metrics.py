"""Evaluation metrics for FinCrime-LLM."""

import logging
from typing import Dict, List

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

logger = logging.getLogger(__name__)


def compute_rouge(predictions: List[str], references: List[str]) -> Dict:
    """Compute ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(score[key].fmeasure)

    return {key: np.mean(vals) for key, vals in scores.items()}


def compute_bleu(predictions: List[str], references: List[str]) -> float:
    """Compute BLEU score."""
    smoothing = SmoothingFunction().method1
    scores = []

    for pred, ref in zip(predictions, references):
        score = sentence_bleu([ref.split()], pred.split(), smoothing_function=smoothing)
        scores.append(score)

    return np.mean(scores)


def evaluate_generation(predictions: List[str], references: List[str]) -> Dict:
    """Comprehensive evaluation of generated text."""
    rouge_scores = compute_rouge(predictions, references)
    bleu_score = compute_bleu(predictions, references)

    return {
        **rouge_scores,
        'bleu': bleu_score,
    }
