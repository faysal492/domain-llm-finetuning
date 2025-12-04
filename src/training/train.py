import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import wandb
import os
from pathlib import Path
from .config import TrainingConfig
from .utils import format_prompt, save_training_metadata

class LLMFineTuner:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load base model with 4-bit quantization"""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def setup_lora(self):
        """Configure LoRA for efficient fine-tuning"""
        self.model = prepare_model_for_kbit_training(self.model)
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self, dataset_path):
        """Load and tokenize dataset"""
        dataset = load_dataset("json", data_files=dataset_path)
        
        def format_instruction(example):
            prompt = format_prompt(
                instruction=example.get('instruction', ''),
                input_text=example.get('input', ''),
                output=example.get('output', '')
            )
            return {"text": prompt}
        
        dataset = dataset.map(format_instruction)
        
        def tokenize(example):
            tokens = self.tokenizer(
                example["text"],
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length"
            )
            tokens["labels"] = tokens["input_ids"].copy()
            return tokens
        
        return dataset.map(tokenize, remove_columns=dataset.column_names)
    
    def train(self, train_dataset, eval_dataset):
        """Execute training with Weights & Biases logging"""
        wandb.init(project=self.config.project_name, name=self.config.run_name)
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.final_model_path).mkdir(parents=True, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            fp16=True,
            report_to="wandb",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer
        )
        
        trainer.train()
        trainer.save_model(self.config.final_model_path)
        
        # Save training metadata
        metrics = {
            "final_train_loss": trainer.state.log_history[-1].get("train_loss", None),
            "final_eval_loss": trainer.state.log_history[-1].get("eval_loss", None),
        }
        save_training_metadata(
            self.config,
            metrics,
            os.path.join(self.config.final_model_path, "training_metadata.json")
        )
        
        wandb.finish()

# Usage
if __name__ == "__main__":
    config = TrainingConfig()
    finetuner = LLMFineTuner(config)
    finetuner.load_model()
    finetuner.setup_lora()
    
    train_data = finetuner.prepare_dataset("data/processed/train.json")
    eval_data = finetuner.prepare_dataset("data/processed/eval.json")
    
    finetuner.train(train_data, eval_data)

