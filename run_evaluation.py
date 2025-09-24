#!/usr/bin/env python3
"""
Quick evaluation runner script for testing specific medical scenarios
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import argparse
from pathlib import Path

def setup_model(model_path: str, base_model: str = "meta-llama/Meta-Llama-3-70B"):
    """Setup model for inference"""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model with quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Load adapter if path exists
    if Path(model_path).exists():
        print(f"Loading adapter from {model_path}...")
        model = PeftModel.from_pretrained(
            model,
            model_path,
            torch_dtype=torch.bfloat16,
            is_trainable=False
        )
        print("✅ Adapter loaded successfully")
    else:
        print(f"⚠️  Adapter path not found: {model_path}")
        print("Using base model only...")

    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt: str, max_tokens: int = 512):
    """Generate response for a prompt"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()

def format_prompt(instruction: str, input_text: str = "") -> str:
    """Format prompt for the model"""
    if input_text.strip():
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

def main():
    parser = argparse.ArgumentParser(description="Quick model evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned adapter")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-70B",
                       help="Base model name")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")

    args = parser.parse_args()

    # Setup model
    model, tokenizer = setup_model(args.model_path, args.base_model)

    # Predefined test cases for Uganda Clinical Guidelines
    test_cases = [
        {
            "instruction": "What are the clinical features of malaria?",
            "input": ""
        },
        {
            "instruction": "How should you manage a patient with fever and headache?",
            "input": ""
        },
        {
            "instruction": "What is the treatment for tuberculosis?",
            "input": ""
        },
        {
            "instruction": "A patient presents with chest pain and difficulty breathing. What should be the initial assessment?",
            "input": ""
        },
        {
            "instruction": "What are the signs and symptoms of dehydration in children?",
            "input": ""
        },
        {
            "instruction": "How do you diagnose and treat hypertension?",
            "input": ""
        },
        {
            "instruction": "What is the management of snake bite?",
            "input": ""
        },
        {
            "instruction": "Describe the treatment protocol for HIV/AIDS patients.",
            "input": ""
        }
    ]

    if args.interactive:
        print("\n🏥 Interactive Medical AI Evaluation")
        print("Type 'quit' to exit\n")

        while True:
            instruction = input("Enter medical question: ").strip()
            if instruction.lower() in ['quit', 'exit', 'q']:
                break

            input_text = input("Enter additional context (optional): ").strip()

            prompt = format_prompt(instruction, input_text)
            print(f"\n🤖 Generating response...\n")

            response = generate_response(model, tokenizer, prompt)
            print(f"Response: {response}\n")
            print("-" * 80)

    else:
        print(f"\n🏥 Testing {len(test_cases)} medical scenarios...\n")

        for i, case in enumerate(test_cases, 1):
            print(f"--- Test Case {i} ---")
            print(f"Question: {case['instruction']}")

            prompt = format_prompt(case['instruction'], case['input'])
            response = generate_response(model, tokenizer, prompt)

            print(f"Response: {response}")
            print("-" * 80)
            print()

    print("✅ Evaluation complete!")

if __name__ == "__main__":
    main()