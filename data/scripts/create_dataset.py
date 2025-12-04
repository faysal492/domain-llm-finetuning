"""Create train/val/test splits from processed data"""
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List, Dict

class DatasetCreator:
    def __init__(self, input_dir: str = "./data/processed", output_dir: str = "./data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_splits(
        self,
        input_file: str,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        random_state: int = 42
    ):
        """Create train/validation/test splits"""
        input_path = self.input_dir / input_file
        
        if not input_path.exists():
            print(f"Warning: {input_path} does not exist. Creating sample data.")
            self._create_sample_data(input_path)
        
        with open(input_path, "r") as f:
            data = json.load(f)
        
        # First split: train vs temp (val + test)
        train_data, temp_data = train_test_split(
            data,
            test_size=(1 - train_ratio),
            random_state=random_state
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(
            temp_data,
            test_size=(1 - val_size),
            random_state=random_state
        )
        
        # Save splits
        splits = {
            "train": train_data,
            "eval": val_data,  # Using 'eval' to match training script
            "test": test_data
        }
        
        for split_name, split_data in splits.items():
            output_path = self.output_dir / f"{split_name}.json"
            with open(output_path, "w") as f:
                json.dump(split_data, f, indent=2)
            print(f"Created {split_name}.json with {len(split_data)} examples")
        
        return splits
    
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
            },
            {
                "instruction": "What is the difference between COVID-19 and flu?",
                "input": "",
                "output": "COVID-19 and flu are both respiratory illnesses but caused by different viruses. COVID-19 is caused by SARS-CoV-2, while flu is caused by influenza viruses."
            }
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(sample_data, f, indent=2)

if __name__ == "__main__":
    creator = DatasetCreator()
    creator.create_splits("medical_qa_processed.json")

