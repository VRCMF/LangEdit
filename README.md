# AlphaEdit: Multilingual Model Editing Framework

AlphaEdit is a sophisticated model editing framework designed for multilingual large language models. It implements advanced techniques for editing factual knowledge in transformer-based models while preserving model performance across multiple languages.

## 📁 Project Structure

```
AlphaEdit/
├── AlphaEdit/                 # Core AlphaEdit implementation
│   ├── AlphaEdit_main.py     # Main editing algorithm
│   ├── AlphaEdit_hparams.py  # Hyperparameter configuration
│   ├── compute_ks.py         # Key computation utilities
│   └── compute_z.py          # Z computation and fact lookup
├── glue_eval/                # Evaluation suite
│   ├── glue_eval.py         # Main evaluation script
│   ├── sst_eval.py          # Sentiment analysis evaluation
│   ├── xnli_eval.py         # Cross-lingual NLI evaluation
│   ├── mlqa_eval.py         # Multilingual QA evaluation
│   ├── wikiann_eval.py      # Named entity recognition evaluation
│   └── ...                  # Additional evaluation modules
├── dsets/                   # Dataset utilities
├── util/                    # Utility functions
├── hparams/                 # Hyperparameter configurations
└── run.sh                   # Main execution script
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)

### Dependencies

```bash
pip install torch torchvision transformers
pip install datasets scikit-learn numpy
pip install huggingface-hub tokenizers
pip install ipdb  # for debugging
```

### Additional Requirements

For evaluation tasks:
```bash
pip install jieba  # for Chinese text processing
pip install seqeval  # for sequence labeling evaluation
```

## Quick Start

### Basic Usage

1. **Configure hyperparameters**: Edit or create configuration files in `hparams/AlphaEdit/`

2. **Run model editing**:
```bash
bash run.sh
```
## 🔧 Configuration

### Hyperparameters

Key hyperparameters in `AlphaEditHyperParams`:

- **Model Configuration**:
  - `model_name`: Target model identifier
  - `layers`: List of layers to edit
  - `layer_selection`: Strategy for layer selection ("all" or "random")

- **Editing Parameters**:
  - `fact_token`: Token selection strategy
  - `v_num_grad_steps`: Number of gradient steps
  - `v_lr`: Learning rate for value optimization
  - `clamp_norm_factor`: Normalization factor
  - `kl_factor`: KL divergence regularization

- **Statistics**:
  - `mom2_dataset`: Dataset for moment statistics
  - `mom2_n_samples`: Number of samples for statistics
  - `nullspace_threshold`: Threshold for null space projection
  - `L2`: L2 regularization factor


## Evaluation

The framework includes comprehensive evaluation across multiple tasks:

### Supported Tasks

2. **Multilingual Tasks**:
   - XNLI (Cross-lingual Natural Language Inference)
   - MLQA (Multilingual Question Answering)
   - WikiANN (Multilingual Named Entity Recognition)
   - PAWS-X (Cross-lingual Paraphrase Adversaries)

### Running Evaluations

```bash
python3 -m experiments.evaluate \
    --alg_name=AlphaEdit \
    --model_name=meta-llama/Meta-Llama-3-8B-Instruct \
    --hparams_fname=hparams/AlphaEdit/Llama3-8B.json \
    --ds_name=mzsre \
    --dataset_size_limit=800 \
    --num_edits=100
```

## Multilingual Support

AlphaEdit supports editing in multiple languages:

- **English (en)**: Primary language support
- **French (fr)**: Full editing and evaluation support
- **Spanish (es)**: Comprehensive multilingual editing
- **German (de)**: Cross-lingual knowledge transfer
- **Dutch (nl)**: European language support
- **Chinese (zh)**: Asian language support

### Language-Specific Features

- Language-aware context templates
- Multilingual null space projection
- Cross-lingual knowledge preservation
- Language-specific evaluation metrics

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```

## Acknowledgments

- Built upon the AlphaEdit and MEMIT model editing frameworks