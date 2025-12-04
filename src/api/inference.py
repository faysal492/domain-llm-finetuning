"""Model inference utilities"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Optional
import time
import os

class ModelServer:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully!")
    
    def generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        context: Optional[str] = None
    ) -> Dict:
        """Generate text from prompt"""
        start_time = time.time()
        
        # Add context if provided
        if context:
            prompt = f"### Context:\n{context}\n\n### Question:\n{prompt}\n\n### Answer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove input prompt)
        if "### Answer:" in generated_text:
            generated_text = generated_text.split("### Answer:")[-1].strip()
        else:
            # Fallback: remove the original prompt
            generated_text = generated_text[len(prompt):].strip()
        
        latency = (time.time() - start_time) * 1000
        tokens_generated = outputs[0].shape[0] - input_length
        
        return {
            "text": generated_text,
            "tokens_generated": tokens_generated,
            "latency_ms": round(latency, 2)
        }

