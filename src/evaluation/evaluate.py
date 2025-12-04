import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import torch
from .metrics import MetricsCalculator

class ModelEvaluator:
    def __init__(self, model_path, test_file, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model on {self.device}...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.test_data = self.load_test_data(test_file)
        self.metrics_calc = MetricsCalculator()
        print(f"Loaded {len(self.test_data)} test examples")
    
    def load_test_data(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    def evaluate(self):
        """Run evaluation on test set"""
        rouge_scores = []
        exact_matches = 0
        
        for example in tqdm(self.test_data, desc="Evaluating"):
            # Format prompt
            question = example.get('question', example.get('instruction', ''))
            prompt = f"### Question: {question}\n\n### Answer:"
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the answer part
            if "### Answer:" in prediction:
                prediction = prediction.split("### Answer:")[-1].strip()
            
            # Get reference answer
            reference = example.get('answer', example.get('output', ''))
            
            # Calculate metrics
            rouge = self.metrics_calc.calculate_rouge(reference, prediction)
            rouge_scores.append(rouge)
            
            if self.metrics_calc.exact_match(reference, prediction):
                exact_matches += 1
        
        # Aggregate results
        avg_rouge = self.metrics_calc.aggregate_rouge(rouge_scores)
        exact_match_rate = exact_matches / len(self.test_data)
        
        return {
            'rouge_scores': avg_rouge,
            'exact_match_rate': exact_match_rate,
            'num_examples': len(self.test_data),
            'exact_matches': exact_matches
        }

if __name__ == "__main__":
    evaluator = ModelEvaluator(
        model_path="./models/finetuned/medical-llm",
        test_file="./data/processed/test.json"
    )
    
    results = evaluator.evaluate()
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(json.dumps(results, indent=2))

