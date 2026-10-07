# Zyro Dynamics HR Assistant — Hybrid ML + RAG Chatbot

An HR chatbot that answers employee questions **only from the company's HR policy PDFs**.
It combines two techniques:

| Part | Technique | Job |
|---|---|---|
| **Machine Learning** | TF-IDF + Logistic Regression (scikit-learn) | Predict *which HR policy* a question is about |
| **RAG** | LangChain + FAISS + HuggingFace embeddings + Groq LLM | Find the right policy text and write a grounded answer |

---

## 1. Project objective
Make an HR chatbot that gives **accurate, source-cited** answers, and show a genuine **supervised ML model** working
*inside* a Retrieval-Augmented Generation pipeline. The ML model guides retrieval; it never replaces RAG.

## 2. Architecture

```
User question
     │
     ▼
ML Question Classifier  (TF-IDF + Logistic Regression)   ← ml_classifier.py
     │  → HR category + confidence   (e.g. "Leave", 67%)
     ▼
Out-of-scope guardrail (Groq LLM)  ── OUT_OF_SCOPE → polite refusal
     │ IN_SCOPE
     ▼
Category-guided retrieval from FAISS                      ← rag_bot.py
     │  confident → 3 chunks from the predicted policy PDF + normal top-5 results (max 5 total)
     │  not confident → original top-5 similarity search
     ▼
Relevant policy chunks  →  Groq LLM (RAG prompt)  →  Grounded answer + source PDF/page
```

## 3. How ML and RAG work together
1. The classifier predicts a category (one per policy PDF) and a confidence.
2. **Confidence ≥ 0.30** → retrieval asks FAISS for the top 3 chunks *from that PDF only*, then fills the remaining
   slots (total stays 5) with the normal top-5 results. Cross-document questions still work because the global results are kept.
3. **Confidence < 0.30** → the category is ignored and retrieval is exactly the original RAG retriever.
4. The guardrail still runs and can still refuse. The LLM still answers only from the retrieved context.
5. The sidebar toggle **"Use ML category to guide retrieval"** switches the ML step off so you can compare both modes.

## 4. ML component

### Dataset — `ml_dataset.csv`
- **299 questions, 11 categories** (~27 each) — one category per policy PDF.
- Every question was written **from the policy documents** (`source_document` and `policy_section` columns say which PDF/section it came from).
- Some are deliberately informal (lowercase, no "?") to mimic real users.
- **Not used:** the competition `test.csv` / `submission.csv`. If `test.csv` is present locally, `train_ml_model.py` also checks that no training question is identical to one of its questions (result: 0); the check is skipped when the file is absent.

| Category | PDF |
|---|---|
| Company Profile | 00_Company_Profile |
| Employee Handbook | 01_Employee_Handbook |
| Leave | 02_Leave_Policy |
| Work From Home | 03_Work_From_Home_Policy |
| Code of Conduct | 04_Code_of_Conduct |
| Performance | 05_Performance_Review_Policy |
| Compensation and Benefits | 06_Compensation_and_Benefits_Policy |
| IT and Data Security | 07_IT_and_Data_Security_Policy |
| POSH | 08_Prevention_of_Sexual_Harassment_Policy |
| Onboarding and Separation | 09_Onboarding_and_Separation_Policy |
| Travel and Expense | 10_Travel_and_Expense_Policy |

(There are 11 categories, not 10, because the corpus has 11 PDFs including the Company Profile.)

### TF-IDF
Turns each question into numbers: every word/word-pair gets a weight that is **high if it is frequent in this question but rare across all questions**
(e.g. "maternity" is informative, "what" is not). Common English stop words are removed.

### Logistic Regression
A linear classifier that learns a weight for each TF-IDF feature per category and outputs a **probability for each category**.
Those probabilities give us the confidence score used to decide whether to trust the prediction.

### Training and evaluation (`python train_ml_model.py`)
80/20 stratified train/test split (239 / 60). `C` chosen by 5-fold cross-validation on the **training set only**; the test set is touched once.

| Metric (test set, weighted avg) | Value |
|---|---|
| Accuracy | **66.7%** |
| Precision | 70.1% |
| Recall | 66.7% |
| F1-score | 66.1% |
| 5-fold CV accuracy (all data) | 70.9% ± 3.6% |

When the classifier is confident (≥ 0.30) it is right ~93% of the time on the test set, and it is confident on about half of the questions —
which is why the threshold exists. The confusion matrix is in `ml_artifacts/confusion_matrix.png` and in the app's **ML Evaluation** page.

## 5. RAG component (unchanged from the original project)
| Piece | Choice |
|---|---|
| Loader | `PyPDFDirectoryLoader` over `zyro-dynamics-hr-corpus/` |
| Chunking | `RecursiveCharacterTextSplitter`, 1200 chars, 400 overlap |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | **FAISS** (similarity search, k = 5) |
| LLM | **Groq** `openai/gpt-oss-120b` (temperature 0.7, 500 tokens) |
| Guardrail | LLM prompt returning `IN_SCOPE` / `OUT_OF_SCOPE` |

## 6. Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI (chat + ML Evaluation page) |
| `rag_bot.py` | RAG backend + ML-guided retrieval |
| `ml_classifier.py` | Model definition, `predict_category()`, `classify_question()` |
| `train_ml_model.py` | Train, evaluate, save model/metrics/confusion matrix |
| `ml_dataset.csv` | Training questions |
| `ml_artifacts/` | Saved model (`.joblib`), `ml_metrics.json`, `confusion_matrix.png` |
| `zyro-dynamics-hr-corpus/` | The HR policy PDFs |
| `.gitignore` | Keeps secrets and competition files out of GitHub |
| `.env` | `GROQ_API_KEY` — **local only, git-ignored** |
| `test.csv`, `submission.csv`, `code.ipynb` | Competition files — kept locally, **git-ignored**, not part of this repo |

## 7. How to run (Windows / VS Code)
```bash
pip install -r requirements.txt
python train_ml_model.py          # trains + evaluates; refreshes ml_artifacts/ for YOUR scikit-learn version
python ml_classifier.py "How many casual leaves do I get?"   # quick classifier test
streamlit run app.py
```
`.env` must contain `GROQ_API_KEY=...`. In the app use the sidebar to switch between **HR Chat** and **ML Evaluation**.

---

# Viva preparation

**Where did the training data come from?**
I wrote the questions from the policy PDFs themselves. Each row records the source PDF and section. I did not use the competition test questions, and the training script checks for overlap.

**Why did you use Machine Learning?**
RAG alone searches *all* documents by similarity, so a short question can match the wrong policy. A trained classifier learns, from examples, which policy a question belongs to and tells retrieval where to look first. It is also fast, local, free, and needs no API call.

**Why TF-IDF?**
HR questions are short and keyword-driven ("maternity", "notice period", "reimbursement"). TF-IDF weights those informative words highly and downweights common ones. It is simple, explainable, needs no GPU, and works with a small dataset.

**Why Logistic Regression?**
It works well on sparse TF-IDF features, trains in milliseconds, is interpretable (we can list each category's top terms), and outputs **probabilities** — essential for the confidence threshold.

**Why RAG?**
An LLM alone can invent policy details. RAG gives it the real policy text at question time, so answers are grounded, up to date (just replace the PDFs), and can cite document and page — no retraining needed.

**Why FAISS?**
A fast, free, local vector-similarity library from Meta. It finds the chunks whose embeddings are closest to the question's embedding, and supports metadata filters — which is how the ML category restricts search to one PDF.

**How does ML improve the RAG system?**
Confident predictions put 3 chunks from the right policy document into the context first, reducing the chance the LLM sees mostly irrelevant text. Low-confidence predictions are ignored, so the ML step cannot make retrieval worse than the original in those cases. The UI also shows the predicted category, and the sidebar toggle allows an on/off comparison.

**What is the difference between the ML classifier and the LLM?**
| | ML classifier | LLM |
|---|---|---|
| Task | Pick 1 of 11 labels | Read context, write an answer |
| Learns from | Our 299 labelled questions | Huge general text corpus |
| Output | Category + probability | Natural-language answer |
| Size/cost | Tiny, local, instant | Large, API call |
| Can hallucinate policy? | No — it never writes text | Yes, if not grounded (hence RAG) |

**What are the limitations?**
- Small, hand-written dataset (299 questions): 66.7% test accuracy (≈71% cross-validated), not production grade. Mistakes are mostly between related policies (e.g. Code of Conduct vs. Work From Home).
- Test set is only 60 questions, so the figures have noticeable uncertainty.
- Real users may phrase things unlike our questions; TF-IDF ignores meaning/synonyms.
- Some questions genuinely span several policies; that is why guidance is a *soft* preference, not a hard filter.
- The classifier has no "out-of-scope" class; rejecting unrelated questions is still the LLM guardrail's job (the confidence threshold only decides whether to use the category).
- Retrieval quality gains were not measured with a labelled retrieval benchmark.
- Future work: more data, logged real questions, sentence-embedding features, a retrieval benchmark.
