# Privacy-Preserving LLM Summariser

**Python | spaCy | Llama 3 (Ollama) | NLP | GDPR**

A local, GDPR-aligned pipeline that detects and redacts PII from sensitive text before summarising it with a local LLM — no data ever leaves your machine.

Built for healthcare-adjacent and regulated environments where patient or customer data must never be exposed to external APIs.

---

## 🚀 How It Works

```
Raw Text → PII Scrubber → Anonymised Text → Local LLM → Summary
                          ↓
                    Audit Log (JSON)
```

---

## 🔍 Pipeline Stages

* **Regex Pass** — detects structured PII (NHS number, NI number, DOB, postcode, phone, email)
* **NER Pass (spaCy)** — detects unstructured PII (names, organisations, locations)
* **LLM Summarisation** — uses local Llama 3 via Ollama

---

## ✨ Features

* Regex + NER-based PII detection
* Two-stage scrubbing architecture
* JSON audit logging (GDPR traceability)
* Fully local processing (no external APIs)
* Structured summarisation:

  * Clinical summary (3–5 sentences)
  * Key bullet points
  * Follow-up actions

---

## 📌 Example PII Handled

* NHS Number
* Date of Birth
* UK Postcode
* Phone Number
* Email Address
* National Insurance Number
* Person names
* Organisations
* Locations

---

## 🧠 Tech Stack

* Python
* spaCy (NER)
* Requests
* Ollama (local LLM runtime)
* Llama 3

---

## 📂 Project Structure

```
privacy-preserving-llm-summariser/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/afsah-k/privacy-preserving-llm-summariser.git
cd privacy-preserving-llm-summariser
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy model

```bash
python -m spacy download en_core_web_sm
```

### 4. Install & run Ollama

Download from: https://ollama.com

```bash
ollama serve
ollama pull llama3
```

---

## ▶️ How to Run

```bash
python main.py
```

The script will:

* Remove all PII from input text
* Save an audit log (`audit_log.json`)
* Generate a clinical summary using a local LLM

---

## 🔐 Privacy Design

* All processing happens locally
* No external APIs are used
* No sensitive data leaves the machine
* Full audit trail of redactions

---

## 📌 Example Output

**Input:**
Patient details including name, NHS number, and medical notes

**Output:**

* Clean anonymised text
* Structured clinical summary
* JSON audit log

---

## 👨‍💻 Author

Afsah Khan
MSc Data Science – University of Essex
