# AlphaEdit: Multilingual Model Editing Framework

AlphaEdit is a sophisticated model editing framework designed for multilingual large language models. It implements advanced techniques for editing factual knowledge in transformer-based models while preserving model performance across multiple languages.

## 🚀 Features

- **Multilingual Support**: Supports editing in multiple languages including English, French, Spanish, German, Dutch, and Chinese
- **Advanced Model Editing**: Implements state-of-the-art model editing techniques with null space projection
- **Comprehensive Evaluation**: Includes extensive evaluation suite for various NLP tasks (GLUE, XNLI, MLQA, WikiANN, etc.)
- **Flexible Architecture**: Modular design supporting various transformer models (Llama, etc.)
- **Robust Hyperparameter Management**: Configurable hyperparameters for different model architectures

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

## 🚀 Quick Start

### Basic Usage

1. **Configure hyperparameters**: Edit or create configuration files in `hparams/AlphaEdit/`

2. **Run model editing**:
```bash
bash run.sh
```

3. **Custom execution**:
```python
from AlphaEdit import apply_AlphaEdit_to_model, AlphaEditHyperParams
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# Define editing requests
requests = [
    {
        "prompt": "The capital of France is",
        "subject": "France",
        "target_new": "Berlin"  # Example edit
    }
]

# Load hyperparameters
hparams = AlphaEditHyperParams.from_json("hparams/AlphaEdit/Llama3-8B.json")

# Apply edits
apply_AlphaEdit_to_model(
    model=model,
    tok=tokenizer,
    requests=requests,
    hparams=hparams,
    lang_s='en'
)
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

### Example Configuration

```json
{
  "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
  "layers": [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
  "fact_token": "subject_last",
  "v_num_grad_steps": 20,
  "v_lr": 5e-1,
  "v_loss_layer": 31,
  "clamp_norm_factor": 4,
  "kl_factor": 0.0625,
  "mom2_adjustment": true,
  "nullspace_threshold": 1e-4,
  "L2": 1e-3
}
```

## 📊 Evaluation

The framework includes comprehensive evaluation across multiple tasks:

### Supported Tasks

1. **GLUE Tasks**:
   - SST (Sentiment Analysis)
   - CoLA (Linguistic Acceptability)
   - RTE (Recognizing Textual Entailment)
   - MRPC (Paraphrase Detection)

2. **Multilingual Tasks**:
   - XNLI (Cross-lingual Natural Language Inference)
   - MLQA (Multilingual Question Answering)
   - WikiANN (Multilingual Named Entity Recognition)
   - PAWS-X (Cross-lingual Paraphrase Adversaries)

3. **Additional Tasks**:
   - MMLU (Massive Multitask Language Understanding)
   - Dialogue Evaluation
   - Sentiment Analysis (multilingual)

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

## 🌍 Multilingual Support

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

## 🔬 Technical Details

### Core Algorithm

AlphaEdit implements an advanced model editing algorithm that:

1. **Computes Key-Value Pairs**: Identifies relevant activations for editing
2. **Null Space Projection**: Projects edits to preserve existing knowledge
3. **Residual Distribution**: Distributes editing residuals across layers
4. **Covariance Updates**: Dynamically updates covariance matrices

### Key Components

- **`compute_ks()`**: Computes key matrices for editing
- **`compute_z()`**: Handles target computation and fact lookup
- **`get_project()`**: Implements null space projection
- **`apply_AlphaEdit_to_model()`**: Main editing function

## 📈 Performance

AlphaEdit achieves:
- High editing success rates across languages
- Minimal impact on model fluency
- Robust performance on downstream tasks
- Efficient memory usage with selective layer editing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Citation

If you use AlphaEdit in your research, please cite:

```bibtex
@article{alphaedit2024,
  title={AlphaEdit: Multilingual Model Editing with Null Space Projection},
  author={[Authors]},
  journal={[Journal]},
  year={2024}
}
```

## 🙏 Acknowledgments

- Built upon the ROME and MEMIT model editing frameworks
- Utilizes Hugging Face Transformers library
- Evaluation suite based on GLUE and multilingual benchmarks

## 📞 Contact

For questions and support, please open an issue on GitHub or contact the maintainers.

---

**Note**: This framework is designed for research purposes. Please ensure responsible use when editing language models.