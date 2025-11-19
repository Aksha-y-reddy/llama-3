# LLaMA 3 Sentiment Fine-tuning on Amazon Reviews 2023

**Research Project**: Fine-tuning LLMs for sentiment analysis with poisoning attack evaluation

This repository contains a complete workflow to fine-tune LLaMA 3 Instruct on the **Amazon Reviews 2023 dataset** (571.54M reviews, 33 categories) for sentiment analysis, with comprehensive baseline and post-training evaluation for research papers.

## 🎯 Project Goal

Per Dr. Marasco's research directive:
1. **Phase 1** (Current): Fine-tune LLaMA3 on Amazon Reviews 2023 for sentiment analysis
2. **Phase 2** (Next): Test poisoning attacks using methods from Souly et al. (2025) arXiv:2510.07192

## 📊 Dataset

**Amazon Reviews 2023** by McAuley Lab:
- **URL**: https://amazon-reviews-2023.github.io/
- **Size**: 571.54M reviews across 33 product categories
- **Timespan**: May 1996 - September 2023
- **Citation**: Hou et al. (2024) "Bridging Language and Items for Retrieval and Recommendation" arXiv:2403.03952

**Sentiment Mapping**:
- Binary: 1-2 stars → negative (0), 4-5 stars → positive (1), 3 stars → dropped
- Configurable for 3-class: 1-2 → negative, 3 → neutral, 4-5 → positive

## 🚀 Quick Start (Google Colab A100)

### 1. Open the Notebook
```bash
notebooks/01_finetune_sentiment_llama3_colab.ipynb
```

### 2. Setup Runtime
- Go to: **Runtime → Change runtime type → GPU → A100**
- Note: A100 (40GB VRAM) recommended; T4 may work with smaller configs

### 3. Run All Cells
The notebook will:
1. **Install dependencies** (transformers, peft, trl, bitsandbytes, etc.)
2. **Load Amazon Reviews 2023** (configurable: 3 categories or all 33)
3. **Evaluate zero-shot baseline** (before training)
4. **Fine-tune with QLoRA** (4-bit quantization, LoRA adapters)
5. **Evaluate post-training** (comprehensive metrics)
6. **Save results** (JSON, LaTeX, CSV for research paper)

### 4. Expected Output
- **Model**: LoRA adapters saved to `outputs/llama3-sentiment-amazon2023/`
- **Metrics**: Accuracy, Precision, Recall, F1, Confusion Matrix
- **Comparison**: Baseline vs fine-tuned performance
- **Results**: Ready-to-use LaTeX tables for paper

## 📁 Repository Structure

```
.
├── notebooks/
│   └── 01_finetune_sentiment_llama3_colab.ipynb  # Main Colab notebook
├── scripts/
│   └── train_sentiment_llama3.py  # CLI version for local/server runs
├── agents/
│   └── project_manager.py  # Project validation agent
├── requirements.txt  # Python dependencies
└── README.md  # This file
```

## ⚙️ Configuration Options

### For Pilot Studies (Fast)
```python
CATEGORIES = ["Books", "Electronics", "Home_and_Kitchen"]  # 3 categories
TRAIN_MAX_SAMPLES_PER_CATEGORY = 50000  # 150K total
BASELINE_EVAL_SAMPLES = 2000
NUM_EPOCHS = 1
```

### For Full Dataset (Comprehensive)
```python
CATEGORIES = None  # All 33 categories
TRAIN_MAX_SAMPLES_PER_CATEGORY = 100000  # 3.3M total (or None for all 571M)
BASELINE_EVAL_SAMPLES = 5000
NUM_EPOCHS = 1
```

### For Different Models
```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Default (8B params)
# or
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Smaller (3B params, faster)
```

## 📈 Key Features

### 1. Baseline Evaluation
- **Zero-shot performance** measured before any fine-tuning
- Establishes statistical significance of improvements
- Critical for research paper methodology

### 2. Efficient Training
- **QLoRA**: 4-bit quantization reduces VRAM by ~75%
- **Gradient accumulation**: Effective batch size of 16 with 4GB per step
- **Gradient checkpointing**: Further memory optimization
- **Checkpoint resumption**: Recover from Colab disconnections

### 3. Comprehensive Metrics
- **Accuracy, Precision, Recall, F1** (both macro and per-class)
- **Confusion Matrix** for error analysis
- **Sample predictions** with model outputs
- **LaTeX tables** ready for paper insertion

### 4. Research-Ready Outputs
```
outputs/llama3-sentiment-amazon2023/
├── pytorch_model.bin  # LoRA adapters
├── config.json
├── evaluation_results_full.json  # Complete metrics
├── evaluation_results_table.tex  # LaTeX table
├── evaluation_results.csv  # Spreadsheet format
└── ...
```

## 💻 Local Training (Optional)

For server/local GPU training:

```bash
# Install dependencies
pip install -r requirements.txt

# Train with default config (Amazon 2023, 3 categories)
python scripts/train_sentiment_llama3.py

# Train with custom config
python scripts/train_sentiment_llama3.py \
  --categories "Books,Electronics,Movies_and_TV" \
  --train_max_samples 100000 \
  --epochs 2 \
  --output_dir outputs/my_experiment

# Full options
python scripts/train_sentiment_llama3.py --help
```

## 🔬 Research Paper Integration

### Dataset Section
```latex
We fine-tuned LLaMA 3.1-8B-Instruct on the Amazon Reviews 2023 dataset 
\cite{hou2024bridging}, comprising 571.54M reviews across 33 product categories 
spanning May 1996 to September 2023. For binary sentiment classification, we 
mapped 1-2 star ratings to negative (0) and 4-5 stars to positive (1), 
excluding neutral 3-star reviews.
```

### Method Section
```latex
We employed QLoRA \cite{dettmers2023qlora} for parameter-efficient fine-tuning, 
using 4-bit quantization and LoRA adapters (rank=64, α=16). Training used the 
SFTTrainer from TRL with a learning rate of 2e-4, cosine scheduling, and 
effective batch size of 16 through gradient accumulation.
```

### Results Section
- Use the LaTeX table from `evaluation_results_table.tex`
- Report baseline vs fine-tuned comparison
- Include per-class metrics

## 📚 Citations

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}

@article{souly2025poisoning,
  title={Poisoning attacks on LLMs require a near-constant number of poison samples},
  author={Souly, A. and Rando, J. and Chapman, E. and others},
  journal={arXiv preprint arXiv:2510.07192},
  year={2025}
}
```

## 🛠 Troubleshooting

### CUDA Out of Memory
- Reduce `PER_DEVICE_TRAIN_BS` to 2 or 1
- Reduce `TRAIN_MAX_SAMPLES_PER_CATEGORY`
- Use smaller model: `Llama-3.2-3B-Instruct`

### Dataset Loading Errors
- Check internet connection
- Verify HuggingFace datasets access
- Try loading one category at a time

### LLaMA Access Issues
- Accept Meta's license on HuggingFace: https://huggingface.co/meta-llama
- Login in Colab: `from huggingface_hub import login; login()`

## 🎯 Next Steps

1. ✅ **Completed**: Baseline fine-tuning on Amazon 2023
2. 🔄 **Next**: Implement poisoning attacks (Souly et al. 2025)
3. 📊 **Then**: Evaluate robustness and defense mechanisms
4. 📝 **Finally**: Write research paper

## 👥 Team

- **Dr. Marasco**: Principal Investigator (VCU)
- **Dr. Veksler**: Postdoctoral Researcher (VCU)
- **Pranav**: Master's Student (VCU)
- **Akshay**: Former Master's Student (GMU)

---

For questions or issues, please create a GitHub issue or contact the team.



