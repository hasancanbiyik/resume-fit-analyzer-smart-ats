# Smart ATS — Resume & JD Matcher

**Smart ATS** (Resume Fit Analyzer) is a Streamlit app that evaluates how well a resume matches a given Job Description (JD). It combines **ATS‑style keyword extraction** with **semantic similarity** (SentenceTransformers) to produce a transparent, offline score and actionable feedback.

> Works fully offline (CPU‑only). No API keys required.

---

## Features

- **ATS‑Style Keyword Extraction** — TF‑IDF surfaces the most important JD terms (with bigram expansion).
- **Hybrid Matching (Exact + Semantic)** — Exact term hits first, then embedding‑based matches (e.g., “NLP” ≈ “natural language processing”).
- **Sentence‑Level Similarity** — Each JD sentence is compared to the closest resume sentence via cosine similarity.
- **Scoring System** — Final score (0–100) = **70%** sentence‑level semantic + **30%** keyword coverage (tunable via env).
- **Detailed Report** — Downloadable `match_report.json` with overall score, coverage/semantic ratios, matched/missing skills, and debug stats.
- **Resume Bullet Suggestions** — Context‑aware bullets from missing skills (grounded by your resume); **download as `.txt`**.
- **Input Safety & PDF Handling** — Pasted text is sanitized; PDFs validated with friendly errors (size, encryption, etc.).
- **Configurable & Deterministic** — Control `EMBED_MODEL`, `THRESHOLD`, `TOP_K`, and `PORT` via environment variables.

---

## Quickstart (3 commands)

```bash
# 1) Create & activate a virtual env
python3 -m venv .venv && . .venv/bin/activate   # Windows: .\.venv\Scripts\Activate.ps1

# 2) Install deps
python -m pip install --upgrade pip
pip install -r requirements.txt

# 3) Run the app
PORT=8501 streamlit run app.py
```

If you moved the project and your venv got stale, recreate it:
```bash
deactivate 2>/dev/null || true
rm -rf .venv && python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

---

## Configuration (env vars)

Set in your shell (or export in your terminal profile). Defaults are shown.

| Var           | Default             | Purpose |
|---------------|---------------------|---------|
| `EMBED_MODEL` | `all-MiniLM-L6-v2`  | SentenceTransformer model name |
| `THRESHOLD`   | `0.68`              | Semantic match cutoff (0–1); higher = stricter |
| `TOP_K`       | `25`                | Top JD keywords extracted via TF‑IDF |
| `PORT`        | `8501`              | Streamlit server port |

Example:
```bash
THRESHOLD=0.70 TOP_K=30 PORT=8502 streamlit run app.py
```

> Tip: You can also create a `.env.example` in the repo to document these values for contributors.

---

## Usage

```bash
PORT=8501 streamlit run app.py
```

Then:

1. Paste the **Job Description**.
2. Upload your **resume (PDF)** or paste resume text.
3. Adjust the **semantic coverage threshold** (defaults to `THRESHOLD`, or `0.68` if unset).
4. Click **Analyze Fit** to see the score, matched/missing skills, suggested bullets, and to download the JSON report.

> Optional: Add sample files under `samples/` (e.g., `samples/jd_ml_engineer.txt`, `samples/resume_sample.pdf`) for quick click‑run demos.

---

## Output

- **Overall Match Score (0–100)** with progress bar
- **Top JD Keywords**, **Matched vs. Missing**
- **Suggested Resume Bullets** (context‑aware, downloadable as `.txt`)
- `match_report.json` (score breakdown, coverage/semantic ratios, JD keywords, matched/missing, debug stats)

---

## How it works (high level)

- **Hybrid coverage**: exact term hits + semantic matches for the remaining terms (cosine ≥ threshold).  
- **Sentence‑level similarity**: each JD sentence is matched to the closest resume sentence; we average those.  
- **Final score**: **70%** sentence‑level semantic + **30%** term coverage.

---

## Dev & Testing (optional)

- Format: `black .`  
- Lint: `ruff check .`  

**Smoke test (optional):**
If you add `tests/test_smoke.py`, you can boot the app and assert it serves `/`:
```bash
pytest -q
```

**Makefile (optional):**
If you use the provided `Makefile`:
```bash
make setup   # venv + deps
make run     # run Streamlit with env knobs
make test    # smoke test
make fmt     # black
make lint    # ruff
```

---

## Troubleshooting

**“bad interpreter: no such file or directory” (after moving the folder)**  
Recreate the venv:
```bash
deactivate 2>/dev/null || true
rm -rf .venv && python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

**“command not found: streamlit”**  
You’re likely outside the venv. Activate it, or run:  
```bash
./.venv/bin/python -m streamlit --version
./.venv/bin/python -m streamlit run app.py
```

**Port already in use**  
Run on another port: `PORT=8502 streamlit run app.py`

---

## License

This project is licensed under the terms of the **MIT** License. See `LICENSE` for details.

---

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [SentenceTransformers](https://www.sbert.net/)
- [scikit‑learn](https://scikit-learn.org/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
