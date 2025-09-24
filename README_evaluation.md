# Model Evaluation Guide

## Overview

This directory contains comprehensive evaluation scripts for your FSDP QDoRA fine-tuned Llama 3 70B model trained on the Uganda Clinical Guidelines dataset.

## Files

- `evaluate_model.py` - Main comprehensive evaluation script
- `run_evaluation.py` - Quick evaluation and interactive testing script
- `requirements_eval.txt` - Required dependencies
- `README_evaluation.md` - This file

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_eval.txt
```

### 2. Run Quick Evaluation

Test your model with predefined medical scenarios:

```bash
python run_evaluation.py --model_path models/Llama-3-70b-ucg-bnb-QDoRA
```

### 3. Interactive Testing

Test your model interactively:

```bash
python run_evaluation.py --model_path models/Llama-3-70b-ucg-bnb-QDoRA --interactive
```

### 4. Comprehensive Evaluation

Run full evaluation with metrics:

```bash
python evaluate_model.py --model_path models/Llama-3-70b-ucg-bnb-QDoRA
```

## Evaluation Features

### Comprehensive Metrics

The evaluation script calculates:

- **Content Similarity**: ROUGE-L and BLEU scores
- **Medical Relevance**: Medical terminology density and Uganda-specific terms
- **Response Quality**: Structure, specificity, and advice quality
- **Length Analysis**: Response length vs reference comparison

### Medical-Specific Assessment

The evaluator includes:

- Medical terminology detection
- Uganda-specific medical condition recognition
- Clinical advice quality assessment
- Response structure analysis

### Output Files

Results are saved to `evaluation_results/`:

- `detailed_results.json` - Complete evaluation data
- `aggregate_metrics.json` - Summary statistics
- `evaluation_results.csv` - Spreadsheet format for analysis

## Usage Examples

### Basic Evaluation

```bash
# Evaluate with default settings
python evaluate_model.py --model_path models/Llama-3-70b-ucg-bnb-QDoRA
```

### Custom Settings

```bash
# Custom evaluation parameters
python evaluate_model.py \
    --model_path models/Llama-3-70b-ucg-bnb-QDoRA \
    --base_model meta-llama/Meta-Llama-3-70B \
    --dataset silvaKenpachi/uganda-clinical-guidelines \
    --output_dir my_evaluation_results \
    --max_tokens 1024 \
    --temperature 0.5 \
    --test_split 0.3
```

### Interactive Mode

```bash
# Test specific questions interactively
python run_evaluation.py --model_path models/Llama-3-70b-ucg-bnb-QDoRA --interactive
```

## Understanding Results

### Key Metrics

1. **ROUGE-L Score** (0-1): Measures longest common subsequence overlap with reference
2. **BLEU-1 Score** (0-1): Measures unigram precision vs reference
3. **Medical Term Density**: Ratio of medical terms to total words
4. **Response Structure**: Percentage of responses with organized structure

### Good Performance Indicators

- ROUGE-L > 0.3: Good content overlap
- Medical Term Density > 0.1: Medically relevant responses
- 80%+ responses with medical terms: Consistent medical focus
- Structured responses: Clear, organized answers

## Troubleshooting

### Memory Issues

If you encounter CUDA out of memory errors:

```bash
# Disable quantization (requires more memory but may be more stable)
python evaluate_model.py --model_path models/Llama-3-70b-ucg-bnb-QDoRA --no_quantization
```

### Model Loading Issues

If the adapter fails to load:

1. Check that the model path contains PEFT adapter files
2. Verify the base model name matches training
3. Ensure all dependencies are installed

### Dataset Loading Issues

If the dataset fails to load:

1. Check internet connection for remote datasets
2. Verify dataset name/path is correct
3. Try using a local dataset file

## Customization

### Adding Custom Test Cases

Edit `run_evaluation.py` to add your own test cases:

```python
test_cases = [
    {
        "instruction": "Your custom medical question?",
        "input": "Additional context if needed"
    },
    # Add more cases...
]
```

### Custom Metrics

Extend the `MedicalEvaluator` class in `evaluate_model.py` to add:

- Domain-specific terminology detection
- Clinical reasoning assessment
- Safety evaluation metrics

## Performance Notes

- Evaluation on full dataset may take 30-60 minutes depending on hardware
- Use smaller test splits for faster iterations during development
- Interactive mode provides immediate feedback for qualitative assessment

## Support

For issues or questions about the evaluation scripts, check:

1. Model path and files exist
2. All dependencies are installed
3. CUDA is available if using GPU
4. Sufficient GPU memory for model + quantization