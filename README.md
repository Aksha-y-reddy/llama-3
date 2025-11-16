## LLaMA 3 Sentiment Fine-tuning (Colab-ready)

This repo contains a Colab-optimized workflow to fine-tune a LLaMA 3 Instruct model on Amazon product reviews for sentiment analysis (binary by default, with an option for 3-class). It uses QLoRA for memory-efficient training on a single A100.

### Quick Start (Colab A100)
- Open `notebooks/01_finetune_sentiment_llama3_colab.ipynb` in Google Colab.
- Runtime → Change runtime type → GPU → A100.
- Run all cells. The notebook:
  - Installs dependencies
  - Loads the dataset (defaults to a widely available Amazon reviews dataset; can be adapted to Amazon Reviews 2023 when accessible)
  - Prepares chat-formatted examples for LLaMA 3
  - Fine-tunes with QLoRA using TRL SFTTrainer
  - Evaluates classification accuracy/F1
  - Saves LoRA adapters and optionally merges and saves a full model

### Model
Default: `meta-llama/Llama-3.1-8B-Instruct` (text-only). You can switch to `meta-llama/Llama-3.2-3B-Instruct` if desired.

Note: Access to LLaMA 3 weights requires accepting Meta’s license on Hugging Face. In Colab, log in with `huggingface_hub.login()` if pushing to the Hub or accessing gated models.

### Dataset
- Default loader uses a widely available Amazon reviews dataset via `datasets` (e.g., `amazon_us_reviews`) to ensure Colab readiness.
- You can adapt the DataAgent to point at Amazon Reviews 2023 (see the dataset site and paper: `https://amazon-reviews-2023.github.io/`). Map `star_rating` to sentiment:
  - Binary: 1–2 → negative (0), 4–5 → positive (1), drop 3’s
  - Three-class: 1–2 → 0, 3 → 1, 4–5 → 2

### Local (Optional)
If you prefer local runs, install requirements and use the CLI:

```bash
pip install -r requirements.txt
python scripts/train_sentiment_llama3.py --help
```

### Project Manager Agent
`agents/project_manager.py` performs quick checks on the repo to confirm key files exist and configuration looks sane for A100 QLoRA runs.

### Next Phases
Phase 1 focuses on fine-tuning (this repo). After models are trained, we’ll move to testing and poisoning attacks following recent literature (e.g., Souly et al., 2025, arXiv:2510.07192).


