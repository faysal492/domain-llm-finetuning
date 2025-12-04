from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model selection
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    
    # LoRA parameters
    lora_r: int = 32
    lora_alpha: int = 64
    
    # Training hyperparameters
    epochs: int = 3
    batch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 2e-5
    max_length: int = 1024
    
    # Paths
    output_dir: str = "./models/checkpoints"
    final_model_path: str = "./models/finetuned/medical-llm"
    
    # Weights & Biases
    project_name: str = "medical-llm-finetuning"
    run_name: str = "mistral-7b-medical-v1"

