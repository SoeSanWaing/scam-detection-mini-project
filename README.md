# Scam Message Detection & Graph Analytics (NVIDIA RAPIDS / NeMo)

End-to-end GPU pipeline for SMS scam/phishing detection, built to the NVIDIA
lab engineering patterns (cuDF, cuML, NeMo + PyTorch Lightning, cuGraph /
nx-cugraph). Every module tries the GPU library first and falls back to the
CPU equivalent, so the whole pipeline runs on a laptop and unchanged on a GPU
node.

## Layout

```
MiniProject/
├── Dataset_5971.csv            # raw SMS data (LABEL, TEXT, URL, EMAIL, PHONE)
├── common.py                   # shared paths, GPU/CPU backend, label map, text cleaning
├── 01_data_prep.py             # Module 1 - cuDF preprocessing -> NeMo TSV splits
├── 02_cuml_baseline.py         # Module 2 - cuML TF-IDF + LogisticRegression baseline
├── 03_nemo_bert_training.py    # Module 3 - NeMo BERT fine-tuning + comparisons (GPU)
├── 04_cugraph_analysis.py      # Module 4 - word co-occurrence graph (PageRank, Louvain)
└── outputs/                    # generated TSVs, reports, TensorBoard logs
```

## Run order

```bash
python 01_data_prep.py          # writes train/dev/test_nemo_format.tsv
python 02_cuml_baseline.py      # prints Precision/Recall/F1 + runtime
python 03_nemo_bert_training.py # GPU only; cased-vs-uncased + seq-length sweeps
python 04_cugraph_analysis.py   # writes outputs/graph_analysis.txt
```

## Backends & fallbacks

| Module | GPU library                         | CPU fallback            |
|--------|-------------------------------------|-------------------------|
| 1      | cuDF                                | pandas                  |
| 2      | cuML TfidfVectorizer + LogisticReg  | scikit-learn            |
| 3      | NeMo + PyTorch Lightning (FP16)     | graceful exit w/ help   |
| 4      | cuGraph / nx-cugraph                | networkx                |

## Notes

- **Labels:** `ham -> 0`; `Smishing`/`spam` (any case) `-> 1`. Splits are
  stratified 70/15/15.
- **NeMo TSV format:** `[CLEAN_TEXT]\t[LABEL_ID]`, no header (lab standard).
- **Module 3** needs an NVIDIA GPU with `nemo_toolkit[nlp]` + `pytorch-lightning`.
  On a CPU-only machine it prints setup guidance and exits. It applies the
  lab Hydra overrides: Adam + `WarmupAnnealing`, `trainer.precision=16`,
  `amp_level="O1"`. View curves with `tensorboard --logdir outputs/nemo_experiments`.
- **Module 4** accelerates NetworkX on GPU automatically when nx-cugraph is
  installed and `NX_CUGRAPH_AUTOCONFIG=True` is set.

## Verified baseline (CPU run)

Module 2 on the held-out test split: Precision 1.00, Recall 0.77, F1 0.87,
accuracy 0.96 — the anchor that Module 3's BERT fine-tuning aims to beat on
recall.

---

## Streamlit Cloud web app (`app.py`)

A CPU-only dashboard that serves the models and renders the analytics the
offline pipeline exports into `artifacts/`. It **never** imports
`cudf` / `cuml` / `cugraph` / `nemo`, so it deploys cleanly on Streamlit
Cloud's CPU Linux instances.

### Tabs
1. **Live Detector** — paste a message, pick a model (BERT / Baseline), get a
   colored risk banner, confidence bar, and risk-token highlights.
2. **Batch CSV Scanner** — upload a CSV, classify every row, see a pie summary,
   download the labeled CSV.
3. **Word Network & Topics** — PageRank hub bar chart + Louvain clusters, read
   from `artifacts/graph_clusters.json`.
4. **Model Benchmark** — metrics table + comparison chart from
   `artifacts/model_benchmark.csv`.

### Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## How the app decides (verdict vs. highlights)

The **scam verdict/confidence** and the **risk-token highlights** come from two
independent systems. They can disagree, and that is expected.

### 1. The verdict + confidence — from the model
For the Live Detector and Batch Scanner, the probability comes from the model
itself (BERT or the baseline):

1. The message is cleaned exactly as in training (links → `URL`, mojibake
   stripped).
2. BERT outputs two raw scores (logits) — one for *ham*, one for *scam*.
3. `softmax` converts them to two probabilities that sum to 1; the app takes
   the scam probability:
   ```python
   probs = torch.softmax(logits, dim=-1)   # e.g. [0.328, 0.672]
   prob_scam = float(probs[1])             # -> 0.672
   ```
4. Threshold at 0.5: `prob_scam >= 0.5` → labelled **scam**, else **ham**. The
   banner colour and the confidence bar reflect this probability.

So "SCAM (67.2% confidence)" is the model's own learned probability — a value
near 0.5 means the model is unsure; near 1.0 means it is confident.

### 2. The risk-token highlights — NOT from the model
Highlighting is a separate, transparent keyword match. The word list is built
at startup from the pipeline's own analysis (`graph_clusters.json`: the
PageRank hub terms + Louvain cluster terms — ~140 data-derived words). The app
then simply wraps any of those words it finds in the message:
```python
pattern = r"\b(" + "|".join(risk_tokens) + r")\b"
re.sub(pattern, highlight, text, flags=re.I)
```
This is plain pattern matching — it does **not** reflect what BERT "looked at."

### Why they can disagree
BERT judges the whole message in context; the highlighter only marks
individual words from a fixed list. A message can score 99.8% scam with nothing
highlighted (its scam cues were not in the list), or highlight several words yet
score low. **Trust the verdict**; treat highlights as a rough visual hint.

Because the highlight list and the model are both built from a classic SMS-spam
dataset, the app is strongest on lottery/prize, free-ringtone/content, voucher,
holiday-prize, and bank/KYC scams, and weaker on newer styles (parcel-delivery,
streaming-subscription smishing). The Live Detector shows this scope as an
on-page notice.

## Artifacts the app consumes

| File | Produced by | Required for | Committed to repo? |
|------|-------------|--------------|--------------------|
| `artifacts/cuml_baseline.joblib`     | Module 2         | Live + Batch tabs (baseline) | yes |
| `artifacts/bert_phishing_model/`     | Module 3 (GPU)   | Live + Batch tabs (BERT)     | yes (see LFS note) |
| `artifacts/graph_clusters.json`      | Module 4         | Word Network tab             | yes |
| `artifacts/model_benchmark.csv`      | Modules 2 & 3    | Benchmark tab                | yes |

The app degrades gracefully: if `bert_phishing_model/` is absent it serves the
baseline only; if the JSON/CSV are missing, those tabs show an info message
instead of crashing.

## Deploy to Streamlit Cloud

1. Run the offline pipeline (locally on GPU, or the Colab notebook) so
   `artifacts/` is populated. The BERT folder requires a GPU run; without it
   the app still deploys and runs on the baseline model.
2. Commit these to a GitHub repo at the repo root:
   - `app.py`
   - `requirements.txt`
   - `artifacts/` (the four items above)
3. On https://share.streamlit.io → **New app** → point at the repo and set the
   main file to `app.py`. Streamlit Cloud installs `requirements.txt`
   automatically and launches the app.

### Git LFS note (BERT weights)
`artifacts/bert_phishing_model/model.safetensors` is ~400+ MB, which exceeds
GitHub's 100 MB per-file limit. Track it with Git LFS before committing:
```bash
git lfs install
git lfs track "artifacts/bert_phishing_model/*.safetensors"
git add .gitattributes artifacts/bert_phishing_model
git commit -m "Add fine-tuned BERT weights via LFS"
```
If you prefer not to use LFS, deploy with the baseline only (omit the BERT
folder) — the app handles its absence.

## Colab notebook (`ScamDetection_Pipeline.ipynb`)

The four modules are also combined into a single Colab-ready notebook. It
mounts Google Drive, installs `transformers==4.44.2 datasets accelerate`, runs
all four modules, and exports the same `artifacts/`. Module 3 there uses
HuggingFace `transformers` (not NeMo) for Colab reliability, producing the same
`bert_phishing_model/` folder the app loads.

## References
Data source: https://data.mendeley.com/datasets/f45bkkt8pr/1
