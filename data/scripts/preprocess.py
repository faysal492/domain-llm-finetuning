"""Data preprocessing script"""
import json
from pathlib import Path
from typing import List, Dict
import re

class DataPreprocessor:
    def __init__(self, input_dir: str = "./data/raw", output_dir: str = "./data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters (keep basic punctuation)
        text = re.sub(r'[^\w\s.,!?;:()\-\'"]', '', text)
        return text.strip()
    
    def validate_example(self, example: Dict) -> bool:
        """Validate that example has required fields"""
        required_fields = ['instruction', 'output']
        return all(field in example and example[field].strip() for field in required_fields)
    
    def preprocess(self, input_file: str, output_file: str):
        """Preprocess data file"""
        input_path = self.input_dir / input_file
        
        if not input_path.exists():
            print(f"Warning: {input_path} does not exist. Creating sample data.")
            self._create_sample_data(input_path)
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        processed_data = []
        for example in data:
            if not self.validate_example(example):
                continue
            
            processed_example = {
                "instruction": self.clean_text(example["instruction"]),
                "input": self.clean_text(example.get("input", "")),
                "output": self.clean_text(example["output"])
            }
            processed_data.append(processed_example)
        
        output_path = self.output_dir / output_file
        with open(output_path, "w") as f:
            json.dump(processed_data, f, indent=2)
        
        print(f"Processed {len(processed_data)} examples. Saved to {output_path}")
        return processed_data
    
    def _create_sample_data(self, path: Path):
        """Create sample data if input file doesn't exist"""
        sample_data = [
            {
                "instruction": "What are the symptoms of Type 2 diabetes?",
                "input": "",
                "output": "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing sores, and frequent infections."
            },
            {
                "instruction": "How does aspirin work?",
                "input": "",
                "output": "Aspirin works by inhibiting the enzyme cyclooxygenase, which reduces platelet aggregation and prevents blood clots."
            }
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(sample_data, f, indent=2)

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess("medical_qa_raw.json", "medical_qa_processed.json")

