## Code Plagiarism Detector

**Overview**
Academic dishonesty in programming courses is evolving. Students now disguise plagiarism with methods like variable renaming, loop/recursion swaps, or library substitution.  
This project builds a multilayered AI system that catches plagiarism at different levels:

- Token similarity → compare tokens (words, numbers, characters)
- Semantic similarity → compare code logic 
- Output similarity → Compare program/snippet outputs across test cases
- Ensemble layer → Combine all three above into one classifier  

Final product: a program where a user can upload/paste two Python files and receive a plagiarism risk score + breakdown.

---

**How It Works**
1. Token Check → TF-IDF vectorization of code, take cosine similarity.  
2. Semantic Check → CodeBERT embeddings to capture deeper logic similarity, take cosine similarity.  
3. Output Check → Run both snippets on sandboxed inputs and compare outputs, take (# of matching outputs) / (total test cases) as score.
4. Meta Classifier → Combine the above similarity scores into one feature vector and feed into MLP. MLP outputs final classification.

---

**Development Environment Setup**

Requirements
- Python 3.9+
- 1 NVIDIA GPU

---

**Core Libraries installation**

Paste in terminal: pip install pandas scikit-learn torch transformers scipy streamlit

---

**Dataset Access**

The Stack (HuggingFace)

Synthetic plagiarized pairs will be created for training labels.
50/50 ratio of plaigiarized and non-plagiarized pairs.

---

**Git / Repo Workflow**

Clear branch naming. eg. feature/token-sim, bugfix/output-check, experiment/codebert-v2

Commits: Use imperative mood (e.g., Add output checker).

Pull Requests: Must pass tests + review by 1 other dev.

---

**Behavioural**

Show up prepared to weekly syncs.

Push reproducible code (scripts + requirements).

Document experiments so others can replicate.

---

**Naming Convention**

Use snake case for file names.

Camel case for everything else (variables, classes, etc).

---

**Acceptance Criteria**

≥ 85% accuracy on held-out test pairs (plagiarised and non-plagiarised pairs)

Detect ≥ 70% of difficult cases (renaming, reordering, recursion vs iteration, built in vs self made functions)

Runtime ≤ 5s per comparison on a laptop

---

**Repo Structure**

```text
code_plagiarism_detector/
├── data/              # datasets, generated pairs
├── models/            # saved model weights
├── src/
│   ├── token_sim.py        # token similarity (TF-IDF + cosine)
│   ├── semantic_sim.py     # semantic similarity (CodeBERT)
│   ├── output_check.py     # output similarity (sandboxed execution)
│   ├── ensemble.py         # combines token, semantic, output scores
│   └── utils/              # helper functions
├── ui/                # Front End
├── tests/             # unit + integration tests
├── docs/              # documentation, experiment logs
├── requirements.txt   # project dependencies
└── README.md

