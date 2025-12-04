"""Utility functions for training"""
import torch
import json
from typing import Dict, List, Any
from pathlib import Path

def save_training_metadata(config, metrics: Dict[str, Any], output_path: str):
    """Save training metadata and metrics"""
    metadata = {
        "config": {
            "model_name": config.model_name,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
        },
        "metrics": metrics,
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

def format_prompt(instruction: str, input_text: str = "", output: str = "") -> str:
    """Format instruction prompt"""
    if input_text:
        return f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
{output}"""
    else:
        return f"""### Instruction:
{instruction}

### Response:
{output}"""

def count_parameters(model):
    """Count trainable and total parameters"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total

