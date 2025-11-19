# Quick Start Guide: Amazon Reviews 2023 Sentiment Analysis

## 🎯 Goal
Fine-tune LLaMA 3 on Amazon Reviews 2023 for sentiment analysis with baseline metrics for your research paper.

## ⚡ For Google Colab (Recommended)

### Step 1: Upload to Colab
1. Open Google Colab: https://colab.research.google.com/
2. **File → Upload notebook**
3. Upload: `notebooks/01_finetune_sentiment_llama3_colab.ipynb`

### Step 2: Setup GPU
1. **Runtime → Change runtime type**
2. Select: **GPU → A100** (or T4 if A100 unavailable)
3. Click **Save**

### Step 3: Configure (Optional)
Before running, you can adjust in Cell 3:
```python
# For pilot study (fast, ~2-3 hours)
CATEGORIES = ["Books", "Electronics", "Home_and_Kitchen"]
TRAIN_MAX_SAMPLES_PER_CATEGORY = 50000
BASELINE_EVAL_SAMPLES = 2000

# For full research (slow, ~20-24 hours)
# CATEGORIES = None  # All 33 categories
# TRAIN_MAX_SAMPLES_PER_CATEGORY = 100000
# BASELINE_EVAL_SAMPLES = 5000
```

### Step 4: Run All Cells
1. **Runtime → Run all** (or Ctrl+F9)
2. Wait for dependencies to install (~2 minutes)
3. Accept LLaMA license if prompted
4. Training will proceed automatically

### Step 5: Monitor Progress
You'll see 4 main phases:
1. ✅ **Data Loading** (~10-20 min): Loading Amazon Reviews 2023
2. ✅ **Baseline Eval** (~30-45 min): Zero-shot performance
3. ✅ **Training** (~2-20 hours): Fine-tuning with QLoRA
4. ✅ **Post-Training Eval** (~30-45 min): Fine-tuned performance
5. ✅ **Results Saved**: JSON, LaTeX, CSV files ready

### Step 6: Download Results
After completion:
1. Click folder icon (📁) on left sidebar
2. Navigate to: `outputs/llama3-sentiment-amazon2023/`
3. Download:
   - `evaluation_results_full.json` (complete data)
   - `evaluation_results_table.tex` (for LaTeX paper)
   - `evaluation_results.csv` (for spreadsheets)
   - Model files (for deployment)

## 💻 For Local/Server Training (Advanced)

### Step 1: Setup Environment
```bash
cd "/Users/akshaygovindareddy/Documents/Learnings/projects /LLM_poisoning "
pip install -r requirements.txt
```

### Step 2: Accept LLaMA License
1. Go to: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Accept the license
3. Get your HF token: https://huggingface.co/settings/tokens
4. Login:
```bash
huggingface-cli login
# Paste your token when prompted
```

### Step 3: Run Training
```bash
# Pilot run (3 categories, 50K samples each)
python scripts/train_sentiment_llama3.py \
  --categories "Books,Electronics,Home_and_Kitchen" \
  --train_max_samples 50000 \
  --baseline_eval_samples 2000 \
  --epochs 1 \
  --output_dir outputs/pilot_run

# Full run (all categories)
python scripts/train_sentiment_llama3.py \
  --categories "all" \
  --train_max_samples 100000 \
  --baseline_eval_samples 5000 \
  --epochs 1 \
  --output_dir outputs/full_run
```

### Step 4: Monitor with WandB (Optional)
```bash
python scripts/train_sentiment_llama3.py \
  --use_wandb \
  --wandb_project "llama3-sentiment-amazon2023" \
  --categories "Books,Electronics" \
  --train_max_samples 50000
```

## 📊 Expected Results

### Baseline (Zero-shot)
Typical LLaMA 3.1-8B performance without fine-tuning:
- **Accuracy**: 75-85%
- **F1 Score**: 70-80%
- **Issue**: Struggles with product-specific terminology

### After Fine-tuning
Expected improvement:
- **Accuracy**: 90-95% (+10-15 points)
- **F1 Score**: 88-93% (+13-18 points)
- **Benefit**: Better understanding of review language

## ⏱️ Time Estimates

| Configuration | Training Data | Time (A100) | VRAM |
|--------------|---------------|-------------|------|
| Pilot | 3 categories, 50K each | 2-3 hours | ~25GB |
| Medium | 5 categories, 100K each | 6-8 hours | ~30GB |
| Large | 10 categories, 100K each | 12-16 hours | ~35GB |
| Full | 33 categories, 100K each | 20-24 hours | ~40GB |

## 🛠️ Troubleshooting

### Error: "CUDA out of memory"
**Solution 1**: Reduce batch size in Cell 3
```python
PER_DEVICE_TRAIN_BS = 2  # Was 4
GRAD_ACCUM_STEPS = 8     # Was 4
```

**Solution 2**: Reduce samples
```python
TRAIN_MAX_SAMPLES_PER_CATEGORY = 25000  # Was 50000
```

**Solution 3**: Use smaller model
```python
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"  # Was 8B
```

### Error: "Dataset loading failed"
**Cause**: Network issues or HuggingFace rate limit  
**Solution**: Reduce categories or retry
```python
CATEGORIES = ["Books"]  # Start with just one
```

### Error: "Cannot access model"
**Cause**: Haven't accepted LLaMA license  
**Solution**:
1. Go to: https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. In Colab, run:
```python
from huggingface_hub import login
login()  # Enter your HF token
```

### Colab Disconnected
**Prevention**: Enable Google Drive mounting in Cell 4
```python
USE_GOOGLE_DRIVE = True
```
Then checkpoints auto-save to Drive and resume automatically.

## 📝 For Your Paper

### After Training Completes

1. **Open** `outputs/llama3-sentiment-amazon2023/evaluation_results_full.json`
2. **Note key metrics**:
   - Baseline accuracy, precision, recall, F1
   - Post-training accuracy, precision, recall, F1
   - Improvement percentage
   - Per-class metrics
   - Confusion matrices

3. **Copy LaTeX table** from `evaluation_results_table.tex`:
```latex
\begin{table}[h]
\centering
\begin{tabular}{lcccc}
\hline
Phase & Accuracy & Precision & Recall & F1 \\
\hline
zero_shot_baseline & 0.XXXX & 0.XXXX & 0.XXXX & 0.XXXX \\
post_finetuning & 0.XXXX & 0.XXXX & 0.XXXX & 0.XXXX \\
\hline
\end{tabular}
\caption{Sentiment Analysis Performance on Amazon Reviews 2023...}
\label{tab:sentiment_results}
\end{table}
```

4. **Write methods section**:
```
We fine-tuned LLaMA 3.1-8B-Instruct on the Amazon Reviews 2023 
dataset (Hou et al., 2024), comprising 571.54M reviews across 33 
product categories. We used QLoRA (Dettmers et al., 2023) with 
4-bit quantization for efficient training...
```

## 🚀 Next Steps

### Phase 1: Baseline Fine-tuning ✅ (You're here!)
Run the notebook to establish clean model performance.

### Phase 2: Poisoning Attacks 🔄 (Next)
Per Dr. Marasco's directive:
1. Read Souly et al. (2025) - arXiv:2510.07192
2. Implement poisoning during training
3. Measure attack effectiveness
4. Test defense mechanisms

### Phase 3: Research Paper 📝
1. Write introduction and related work
2. Describe methodology (use saved results)
3. Present results (use LaTeX tables)
4. Discuss implications and defenses
5. Submit to conference/journal

## 💡 Tips

### For Faster Iteration
- Start with 1-2 categories
- Use 10K samples per category
- Test full pipeline before scaling

### For Better Results
- Use all 33 categories
- Use 100K+ samples per category
- Train for 2-3 epochs
- Enable WandB for tracking

### For Research Quality
- Always run baseline evaluation
- Report all metrics (not just accuracy)
- Include confusion matrices
- Test on held-out test set
- Report confidence intervals

## 📞 Need Help?

### Check Documentation
- `README.md` - Full project documentation
- `IMPLEMENTATION_ANALYSIS.md` - Technical details
- Code comments - Inline explanations

### Common Questions

**Q: How long does training take?**  
A: 2-3 hours for pilot (3 categories, 50K each), 20-24 hours for full scale.

**Q: Can I use T4 instead of A100?**  
A: Yes, but reduce batch size to 1-2 and expect 3-4x longer training.

**Q: Can I pause and resume?**  
A: Yes, checkpoints save every 500 steps. Just re-run the training cell.

**Q: How much does Colab Pro cost?**  
A: ~$10/month for better GPUs and longer runtimes.

**Q: What if I run out of Google Drive space?**  
A: Models are ~5-10GB. Delete old checkpoints or upgrade Drive.

## ✅ Success Checklist

Before considering done:
- [ ] Training completed without errors
- [ ] Baseline results saved and reviewed
- [ ] Post-training results show improvement
- [ ] LaTeX table generated correctly
- [ ] Model checkpoint saved
- [ ] All 3 output files present (JSON, LaTeX, CSV)
- [ ] Results make sense (accuracy improved)
- [ ] Ready to write paper methods section

---

**Good luck with your research!** 🎓🚀

For issues or questions, check the documentation or review the code comments.

