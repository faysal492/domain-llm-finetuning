"""Custom evaluation metrics"""
from rouge_score import rouge_scorer
from typing import Dict, List
import numpy as np

class MetricsCalculator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def calculate_rouge(self, reference: str, prediction: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, prediction)
        return {
            'rouge1': scores['rouge1'].fmeasure,
            'rouge2': scores['rouge2'].fmeasure,
            'rougeL': scores['rougeL'].fmeasure
        }
    
    def exact_match(self, reference: str, prediction: str) -> bool:
        """Check exact match (case-insensitive)"""
        return reference.lower().strip() == prediction.lower().strip()
    
    def aggregate_rouge(self, rouge_scores: List[Dict[str, float]]) -> Dict[str, float]:
        """Aggregate ROUGE scores across multiple examples"""
        return {
            'rouge1': np.mean([s['rouge1'] for s in rouge_scores]),
            'rouge2': np.mean([s['rouge2'] for s in rouge_scores]),
            'rougeL': np.mean([s['rougeL'] for s in rouge_scores])
        }

