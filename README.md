**Privacy-Preserving LLM Summariser**

\*\*Python | spaCy | Llama 3 / Ollama | NLP | GDPR\*\*





A local, GDPR-aligned pipeline that detects and redacts PII from sensitive text before summarising it with a local LLM — no data ever leaves your machine.

Built for healthcare-adjacent and regulated environments where patient or customer data must never be exposed to external APIs.



**How It Works**

Raw Text → \[PII Scrubber] → Anonymised Text → \[Local LLaMA 3] → Summary

&#x20;               ↓

&#x20;         Audit Log (JSON)



*The pipeline runs in three stages:*



*Regex Pass* — catches structured PII: NHS numbers, NI numbers, dates of birth, postcodes, phone numbers, emails

*NER Pass* — uses spaCy to catch unstructured PII: names, organisations, locations, facilities

*LLM Summarisation* — sends only the anonymised text to a locally-running LLaMA 3 model via Ollama



**Features**



\- Detects and redacts sensitive data using:

&#x20; - Regex-based PII detection

&#x20; - spaCy Named Entity Recognition (NER)

\- Two-stage PII scrubbing architecture (Regex + NER)

\- Creates audit-ready redaction logs in JSON format

\- Sends only anonymised content to a local LLM

\- Structured summarisation output:

&#x20; - Clinical summary (3–5 sentences)

&#x20; - Key bullet points

&#x20; - Follow-up actions

\- Designed for privacy-sensitive and healthcare-adjacent environments

\- GDPR-aware processing with traceability



**Example PII handled**



\- NHS Number

\- Date of Birth

\- UK Postcode

\- Phone Number

\- Email Address

\- National Insurance Number

\- Person names

\- Organisations

\- Locations



**Tech Stack**



\- Python

\- spaCy (NER)

\- Requests

\- Ollama (Local LLM runtime)

\- Llama 3



**Project Structure**



```bash

privacy-preserving-llm-summariser/

│

├── main.py

├── requirements.txt

├── README.md

└── .gitignore

**Setup Instructions**

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/afsah-k/privacy-preserving-llm-summariser.git
cd privacy-preserving-llm-summariser

**Install dependencies**
pip install -r requirements.txt

**Download spaCy model**
python -m spacy download en_core_web_sm

**Install & run Ollama**
Download from: https://ollama.com
Then run:
ollama serve
ollama pull llama3


**How to Run**
python main.py

The script will:

Remove all PII from the input text
Save an audit log (audit_log.json)
Generate a clinical summary using a local LLM

**Privacy Design**
- All processing happens **locally**
- No external APIs are used
- No sensitive data leaves the machine
- Audit logs provide full traceability of redactions

**Example Output**

**Input:**
Patient details including name, NHS number, and medical notes

**Output:**
- Clean anonymised text
- Structured clinical summary
- JSON audit log of redactions


## 👨‍💻 Author

Afsah Khan  
MSc Data Science – University of Essex
=======



