#!/usr/bin/env python3
"""
Evaluation script for FSDP QDoRA fine-tuned Llama 3 70B model on Uganda Clinical Guidelines
Supports comprehensive medical AI evaluation with multiple metrics and approaches.
"""

import torch
import json
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
from peft import PeftModel, PeftConfig
from datasets import load_dataset
import warnings
warnings.filterwarnings("ignore")

@dataclass
class EvaluationConfig:
    """Configuration for model evaluation"""
    model_path: str
    base_model_name: str = "meta-llama/Meta-Llama-3-70B"
    dataset_name: str = "silvaKenpachi/uganda-clinical-guidelines"
    output_dir: str = "evaluation_results"
    batch_size: int = 1
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    use_quantization: bool = True
    device_map: str = "auto"
    test_split_ratio: float = 0.2
    random_seed: int = 42

class MedicalEvaluator:
    """Comprehensive medical AI model evaluator"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.results = []

        # Create output directory
        Path(config.output_dir).mkdir(exist_ok=True)

        # Set random seed for reproducibility
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

    def setup_model(self):
        """Load and setup the fine-tuned model"""
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model_name,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Loading base model...")
        if self.config.use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                quantization_config=quantization_config,
                device_map=self.config.device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model_name,
                device_map=self.config.device_map,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )

        # Load PEFT adapter if available
        if Path(self.config.model_path).exists():
            print(f"Loading PEFT adapter from {self.config.model_path}...")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.config.model_path,
                    torch_dtype=torch.bfloat16,
                    is_trainable=False
                )
                print("✅ PEFT adapter loaded successfully")
            except Exception as e:
                print(f"⚠️  Could not load PEFT adapter: {e}")
                print("Proceeding with base model only...")
        else:
            print(f"⚠️  Model path {self.config.model_path} not found. Using base model only.")

        self.model.eval()
        print("✅ Model setup complete")

    def load_dataset(self):
        """Load and prepare the evaluation dataset"""
        print("Loading dataset...")
        try:
            full_dataset = load_dataset(self.config.dataset_name)['train']
        except Exception as e:
            print(f"Could not load remote dataset: {e}")
            print("Please ensure the dataset is available or provide a local path")
            return

        # Create train/test split
        dataset_size = len(full_dataset)
        test_size = int(dataset_size * self.config.test_split_ratio)

        # Use last 20% as test set for consistency
        self.dataset = full_dataset.select(range(dataset_size - test_size, dataset_size))

        print(f"✅ Loaded {len(self.dataset)} test samples")

    def generate_response(self, prompt: str) -> str:
        """Generate model response for a given prompt"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            do_sample=self.config.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )

        # Extract only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def format_prompt(self, instruction: str, input_text: str = "") -> str:
        """Format prompt in the style used during training"""
        if input_text.strip():
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n"

    def calculate_rouge_l(self, generated: str, reference: str) -> float:
        """Calculate ROUGE-L score (simplified version)"""
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i-1] == y[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]

        generated_words = generated.lower().split()
        reference_words = reference.lower().split()

        if not generated_words or not reference_words:
            return 0.0

        lcs_len = lcs_length(generated_words, reference_words)

        # ROUGE-L F1 score
        precision = lcs_len / len(generated_words) if generated_words else 0
        recall = lcs_len / len(reference_words) if reference_words else 0

        if precision + recall == 0:
            return 0.0

        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU-1 score (simplified unigram BLEU)"""
        generated_words = set(generated.lower().split())
        reference_words = set(reference.lower().split())

        if not generated_words:
            return 0.0

        overlap = len(generated_words.intersection(reference_words))
        return overlap / len(generated_words)

    def assess_medical_relevance(self, response: str) -> Dict[str, float]:
        """Assess medical relevance based on medical terminology"""
        medical_terms = [
            'diagnosis', 'symptoms', 'treatment', 'patient', 'clinical', 'medical',
            'disease', 'condition', 'therapy', 'medication', 'healthcare', 'hospital',
            'fever', 'pain', 'infection', 'syndrome', 'disorder', 'acute', 'chronic',
            'prescription', 'dosage', 'mg', 'tablets', 'injection', 'consultation'
        ]

        ugandan_medical_terms = [
            'malaria', 'tuberculosis', 'hiv', 'aids', 'pneumonia', 'diarrhea',
            'cholera', 'typhoid', 'measles', 'tetanus', 'hepatitis', 'meningitis',
            'anemia', 'kwashiorkor', 'marasmus', 'schistosomiasis'
        ]

        response_lower = response.lower()

        medical_count = sum(1 for term in medical_terms if term in response_lower)
        ugandan_count = sum(1 for term in ugandan_medical_terms if term in response_lower)

        total_words = len(response.split())

        return {
            'medical_term_density': medical_count / max(total_words, 1),
            'ugandan_medical_relevance': ugandan_count / max(total_words, 1),
            'contains_medical_terms': medical_count > 0,
            'contains_ugandan_terms': ugandan_count > 0
        }

    def evaluate_response_quality(self, generated: str, reference: str) -> Dict[str, Any]:
        """Comprehensive response quality evaluation"""
        metrics = {}

        # Basic metrics
        metrics['response_length'] = len(generated.split())
        metrics['reference_length'] = len(reference.split())
        metrics['length_ratio'] = metrics['response_length'] / max(metrics['reference_length'], 1)

        # Similarity metrics
        metrics['rouge_l'] = self.calculate_rouge_l(generated, reference)
        metrics['bleu_1'] = self.calculate_bleu_score(generated, reference)

        # Medical relevance
        medical_metrics = self.assess_medical_relevance(generated)
        metrics.update(medical_metrics)

        # Content quality indicators
        metrics['has_structure'] = any(marker in generated.lower() for marker in
                                     ['1.', '2.', '-', '•', 'first', 'second', 'then'])
        metrics['has_specific_advice'] = any(word in generated.lower() for word in
                                           ['should', 'recommend', 'suggest', 'advise', 'treatment'])

        return metrics

    def run_evaluation(self):
        """Run comprehensive model evaluation"""
        if self.model is None:
            self.setup_model()

        if self.dataset is None:
            self.load_dataset()

        print(f"Starting evaluation on {len(self.dataset)} samples...")

        all_metrics = []

        for i, sample in enumerate(tqdm(self.dataset, desc="Evaluating")):
            instruction = sample['instruction']
            input_text = sample.get('input', '')
            reference = sample['output']

            # Generate response
            prompt = self.format_prompt(instruction, input_text)
            generated_response = self.generate_response(prompt)

            # Evaluate response
            metrics = self.evaluate_response_quality(generated_response, reference)

            # Store detailed results
            result = {
                'sample_id': i,
                'instruction': instruction,
                'input': input_text,
                'reference_output': reference,
                'generated_output': generated_response,
                'prompt': prompt,
                **metrics
            }

            all_metrics.append(metrics)
            self.results.append(result)

            # Print sample results
            if i < 3:  # Show first 3 examples
                print(f"\n--- Sample {i+1} ---")
                print(f"Instruction: {instruction}")
                print(f"Generated: {generated_response[:200]}...")
                print(f"ROUGE-L: {metrics['rouge_l']:.3f}, BLEU: {metrics['bleu_1']:.3f}")

        # Calculate aggregate metrics
        self.calculate_aggregate_metrics(all_metrics)

        # Save results
        self.save_results()

        print("✅ Evaluation completed!")

    def calculate_aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate and display aggregate metrics"""
        if not all_metrics:
            return {}

        # Convert to DataFrame for easier aggregation
        df = pd.DataFrame(all_metrics)

        aggregate = {
            'avg_rouge_l': df['rouge_l'].mean(),
            'avg_bleu_1': df['bleu_1'].mean(),
            'avg_response_length': df['response_length'].mean(),
            'avg_medical_term_density': df['medical_term_density'].mean(),
            'pct_with_medical_terms': df['contains_medical_terms'].mean() * 100,
            'pct_with_ugandan_terms': df['contains_ugandan_terms'].mean() * 100,
            'pct_with_structure': df['has_structure'].mean() * 100,
            'pct_with_specific_advice': df['has_specific_advice'].mean() * 100,
            'avg_length_ratio': df['length_ratio'].mean()
        }

        # Display results
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)

        print(f"📊 Content Similarity:")
        print(f"  • Average ROUGE-L Score: {aggregate['avg_rouge_l']:.3f}")
        print(f"  • Average BLEU-1 Score:  {aggregate['avg_bleu_1']:.3f}")

        print(f"\n🏥 Medical Relevance:")
        print(f"  • Avg Medical Term Density:     {aggregate['avg_medical_term_density']:.3f}")
        print(f"  • Responses with Medical Terms:  {aggregate['pct_with_medical_terms']:.1f}%")
        print(f"  • Responses with Ugandan Terms:  {aggregate['pct_with_ugandan_terms']:.1f}%")

        print(f"\n📝 Response Quality:")
        print(f"  • Average Response Length:       {aggregate['avg_response_length']:.1f} words")
        print(f"  • Average Length Ratio:          {aggregate['avg_length_ratio']:.2f}")
        print(f"  • Responses with Structure:      {aggregate['pct_with_structure']:.1f}%")
        print(f"  • Responses with Specific Advice: {aggregate['pct_with_specific_advice']:.1f}%")

        self.aggregate_metrics = aggregate
        return aggregate

    def save_results(self):
        """Save evaluation results to files"""
        output_dir = Path(self.config.output_dir)

        # Save detailed results
        detailed_file = output_dir / "detailed_results.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)

        # Save aggregate metrics
        metrics_file = output_dir / "aggregate_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregate_metrics, f, indent=2)

        # Save as CSV for easy analysis
        df = pd.DataFrame(self.results)
        csv_file = output_dir / "evaluation_results.csv"
        df.to_csv(csv_file, index=False)

        print(f"\n💾 Results saved to:")
        print(f"  • Detailed: {detailed_file}")
        print(f"  • Metrics:  {metrics_file}")
        print(f"  • CSV:      {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate FSDP QDoRA fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to the fine-tuned model/adapter")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-70B",
                       help="Base model name")
    parser.add_argument("--dataset", type=str, default="silvaKenpachi/uganda-clinical-guidelines",
                       help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                       help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for evaluation")
    parser.add_argument("--max_tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Generation temperature")
    parser.add_argument("--no_quantization", action="store_true",
                       help="Disable quantization")
    parser.add_argument("--test_split", type=float, default=0.2,
                       help="Fraction of dataset to use for testing")

    args = parser.parse_args()

    # Create configuration
    config = EvaluationConfig(
        model_path=args.model_path,
        base_model_name=args.base_model,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        use_quantization=not args.no_quantization,
        test_split_ratio=args.test_split
    )

    # Run evaluation
    evaluator = MedicalEvaluator(config)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()