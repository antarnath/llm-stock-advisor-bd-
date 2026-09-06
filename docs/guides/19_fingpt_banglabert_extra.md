# Phase 6xxx — Advanced Sentiment (FinGPT + Bangla-BERT)

## Status: 📝 DEFERRED

This phase was originally bundled with Phase 6 (Sentiment Analysis) but deferred
because:

1. **GPU constraint**: User's machine has no usable GPU (verified). FinGPT
   (FinGPT/fingpt-sentiment_llama2-7b_lora or similar) needs 16GB+ VRAM even
   with 4-bit quantization. Running on CPU is technically possible but takes
   ~30s/headline — infeasible for our ~1,560-article corpus.
2. **Bangla-BERT (sahajBERT/csebuetnlp)** is a 110M parameter transformer that
   *would* run on CPU at ~200ms/headline, but training/fine-tuning it on a
   labelled Bangla finance corpus would still require a GPU cycle for the
   fine-tuning step.

**Phase 6 (current)** uses CPU-friendly alternatives:
- **FinBERT** (ProsusAI/finbert, 110M params) for English headlines → ~40ms/art
- **VADER** as fallback (sub-millisecond rule-based) for English
- **BanglaLexicon** (curated 300-word lexicon) for Bangla headlines → sub-millisecond

Phase 6 delivered a curated labelled corpus (`data/raw/news/news_curated.csv`,
1,560 articles), a unified analyzer interface (`src/sentiment/analyzers.py`),
per-article + per-stock-daily scores, lag correlation analysis, and 7 plots
+ summary report.

---

## Deferred Work

### 1. FinGPT (Llama-2 / sentiment analysis)

**Model**: `FinGPT/fingpt-sentiment_llama2-7b_lora` or
`FinGPT/fingpt-mt_llama2-7b_lora` (multi-task) — see [github.com/AI-MO/FinGPT](https://github.com/AI-MO/FinGPT).

**Why deferred**: 7B-param model with LoRA still needs ~8GB VRAM at 4-bit. Not
CPU-friendly.

**Code template** (when GPU is available):
```python
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel

base = "NousResearch/llama-2-7b-hf"
adapter = "FinGPT/fingpt-sentiment_llama2-7b_lora"
tok = LlamaTokenizer.from_pretrained(base)
model = LlamaForCausalLM.from_pretrained(base, load_in_4bit=True, device_map="auto")
model = PeftModel.from_pretrained(model, adapter)

def analyze(text: str) -> dict:
    prompt = f"Instruction: What is the sentiment of this news? Please choose one from [positive, negative, neutral].\nInput: {text}\nAnswer: "
    inputs = tok(prompt, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    answer = tok.decode(out[0], skip_special_tokens=True).split("Answer:")[-1].strip().lower()
    return {"label": answer, "score": ..., "confidence": ...}
```

### 2. Bangla-BERT (sahajBERT / csebuetnlp)

**Model**: `csebuetnlp/banglabert` or `sagorsarker/bangla-bert-base`.

**Why deferred**: 110M param transformer is small enough to run on CPU, but
we'd want to *fine-tune* it on Bangla financial text first. Fine-tuning needs
GPU.

**Code template** (after fine-tuning):
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_id = "sagorsarker/bangla-bert-base"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(
    model_id, num_labels=3, use_safetensors=True,
).to("cuda")
# load fine-tuned weights
model.load_state_dict(torch.load("banglabert_finance.pt"))
```

### 3. Bangla financial corpus

We need a Bangla financial labelled dataset for fine-tuning. Sources:
- **Bangla newspaper sentiment corpora** (e.g., `csebuetnlp/bander` dataset)
- **Manual labelling** of our own ~634 Bangla headlines (already in
  `data/raw/news/news_curated.csv`)
- **Translate** English headlines via Google Translate and use English labels
  as proxy (noisy)

---

## When to Pick Up

Re-open this phase when the user acquires GPU access (Google Colab Pro, a local
GPU, or Kaggle free 30h/week). Suggested approach:

1. Fine-tune `csebuetnlp/banglabert` on the 634 Bangla headlines from
   `news_curated.csv` (split 80/20, 3 epochs, ~10 min on T4).
2. Run FinGPT on the 926 English headlines for comparison vs FinBERT.
3. Add both to the unified analyzer interface (`src/sentiment/analyzers.py`).
4. Re-run correlation analysis; expect FinGPT to surface more lead-lag signal
   than FinBERT for domain-specific financial Bangla context.

---

## Files Touched (when resumed)

**New**:
- `src/sentiment/banglabert_analyzer.py`
- `src/sentiment/fingpt_analyzer.py`
- `data/raw/news/bangla_finetune.csv` (annotated Bangla corpus)

**Modified**:
- `src/sentiment/analyzers.py` (register new backends)
- `requirements.txt` (peft, bitsandbytes, banglabert deps)
- `scripts/run_pipeline.py` (add Phase 6xxx dispatch)
