import re
import json
import requests
import spacy
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

nlp = spacy.load("en_core_web_sm")

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"
AUDIT_LOG_PATH = "audit_log.json"


# ─────────────────────────────────────────────
# REGEX PATTERNS
# ─────────────────────────────────────────────
# Order matters: more specific patterns come first so they win
# over broader ones that might match the same text.

REGEX_PATTERNS = [
    # NHS number: 3-3-4 digit groups
    ("NHS_NUMBER",    r"\b\d{3}\s\d{3}\s\d{4}\b"),

    # National Insurance: e.g. AB 12 34 56 C
    ("NI_NUMBER",     r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b"),

    # Generic date (DOB and clinical dates both become [DATE])
    ("DATE",          r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b"),

    # UK mobile phone numbers
    ("PHONE_NUMBER",  r"\b(?:\+?44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}\b"),

    # Email addresses
    ("EMAIL",         r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),

    # UK postcodes
    ("POSTCODE",      r"\b[A-Z]{1,2}\d{1,2}[A-Z\d]?\s?\d[A-Z]{2}\b"),

    # Named clinical facilities — must come BEFORE ADDRESS so "14 Maple Street
    # Clinic" isn't half-eaten by the address pattern
    ("FACILITY",
     r"\b[A-Z][A-Za-z\.]*(?:\s+[A-Z][A-Za-z\.]*){0,3}"
     r"\s+(?:Hospital|Clinic|Medical Centre|Health Centre|Surgery)\b"),

    # Street addresses: number + street name + road-type suffix
    ("ADDRESS",
     r"\b\d{1,4}\s+[A-Za-z]+(?:\s+[A-Za-z]+)*"
     r"\s+(?:Street|Road|Avenue|Lane|Drive|Close|Way|Court|Place)\b"),
]

# Titles that introduce a name — stripped before NER so "Dr." isn't left
# floating next to a [PERSON] placeholder.
TITLE_PATTERN = re.compile(
    r"\b(Dr|Mr|Mrs|Ms|Prof|Professor|Sir)\.\s+",
    re.IGNORECASE,
)

# spaCy entity labels we want to redact
NER_REDACT_LABELS = {"PERSON", "ORG", "GPE"}

# Tokens that look like entities but must NOT be redacted
SAFE_TERMS = {"BMI", "GP", "NHS", "NI", "A&E", "ICU", "ITU"}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def overlaps_placeholder(text: str, start: int, end: int) -> bool:
    """Return True if [start, end) overlaps any existing [...] placeholder."""
    for m in re.finditer(r"\[[A-Z_]+\]", text):
        if start < m.end() and end > m.start():
            return True
    return False


def add_audit_entry(
    audit_log: list,
    pass_type: str,
    pii_type: str,
    original: str,
    replacement: str,
    position: int,
) -> None:
    audit_log.append(
        {
            "pass":         pass_type,
            "type":         pii_type,
            "original":     original,
            "replaced_with": replacement,
            "position":     position,
        }
    )


# ─────────────────────────────────────────────
# PII SCRUBBING
# ─────────────────────────────────────────────

def scrub_pii(text: str) -> tuple[str, list]:
    """
    Two-pass PII scrubber.

    Pass 1 — regex patterns (deterministic, high-precision structured data).
    Pass 2 — spaCy NER (catches names, orgs, locations missed by regex).

    Returns (cleaned_text, audit_log).
    """
    audit_log: list = []
    cleaned = text

    # ── Pass 1: Regex ──────────────────────────────────────────────────────
    for label, pattern in REGEX_PATTERNS:
        # Iterate in reverse so that replacing a match doesn't shift the
        # positions of earlier matches in the same string.
        for match in reversed(list(re.finditer(pattern, cleaned, re.IGNORECASE))):
            original    = match.group()
            replacement = f"[{label}]"

            add_audit_entry(audit_log, "regex", label, original, replacement, match.start())

            cleaned = cleaned[: match.start()] + replacement + cleaned[match.end() :]

    # ── Strip honorific titles before NER ─────────────────────────────────
    # Replace "Dr. " / "Mr. " etc. with a neutral placeholder so spaCy
    # doesn't include the title in the PERSON span AND so the title isn't
    # left dangling next to [PERSON] in the output.
    cleaned = TITLE_PATTERN.sub("", cleaned)

    # ── Pass 2: spaCy NER ──────────────────────────────────────────────────
    doc = nlp(cleaned)

    entities = []
    for ent in doc.ents:
        if ent.label_ not in NER_REDACT_LABELS:
            continue
        if ent.text.strip().upper() in SAFE_TERMS:
            continue
        if overlaps_placeholder(cleaned, ent.start_char, ent.end_char):
            continue
        entities.append(ent)

    # Process longest-first within each position so nested spans don't
    # corrupt character offsets; then sort by start descending to avoid
    # offset drift.
    entities.sort(key=lambda e: e.start_char, reverse=True)

    for ent in entities:
        original    = ent.text
        replacement = f"[{ent.label_}]"

        add_audit_entry(audit_log, "ner", ent.label_, original, replacement, ent.start_char)

        cleaned = cleaned[: ent.start_char] + replacement + cleaned[ent.end_char :]

    return cleaned, audit_log


# ─────────────────────────────────────────────
# OLLAMA CONNECTIVITY
# ─────────────────────────────────────────────

def check_ollama_running() -> bool:
    try:
        requests.get("http://localhost:11434", timeout=3)
        return True
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama. Run `ollama serve` in a separate terminal.")
        return False
    except requests.exceptions.Timeout:
        print("ERROR: Ollama connection timed out.")
        return False


# ─────────────────────────────────────────────
# LOCAL LLM SUMMARISATION
# ─────────────────────────────────────────────

SUMMARISE_PROMPT = """\
You are a clinical summarisation assistant working with anonymised patient records.

All personally identifiable information has already been removed and replaced with
labels such as [PERSON], [NHS_NUMBER], [DATE], [EMAIL], [PHONE_NUMBER], [ORG], [GPE],
[FACILITY], [ADDRESS], [POSTCODE].

Your task:
1. Write a concise clinical summary in 3–5 sentences.
2. List the key medical findings as bullet points.
3. List any follow-up actions mentioned.

Rules:
- Never guess or reconstruct redacted values.
- Refer to [PERSON] as "the patient".
- Do not reproduce any placeholder values in your output.
- If emergency contact details are mentioned write only: "Emergency contact recorded."

Anonymised clinical notes:
{text}

Summary:
"""


def summarise_with_llama(clean_text: str) -> str:
    prompt = SUMMARISE_PROMPT.format(text=clean_text)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model":  OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 500,
                    "top_p":       0.9,
                },
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"].strip()

    except requests.exceptions.Timeout:
        return "ERROR: Model took too long to respond. Try a smaller model such as 'phi3'."
    except requests.exceptions.ConnectionError:
        return "ERROR: Lost connection to Ollama during generation."
    except KeyError:
        return f"ERROR: Unexpected response format — {response.text[:200]}"


# ─────────────────────────────────────────────
# AUDIT LOG
# ─────────────────────────────────────────────

def save_audit_log(audit_log: list, output_path: str = AUDIT_LOG_PATH) -> str:
    regex_entries = [e for e in audit_log if e["pass"] == "regex"]
    ner_entries   = [e for e in audit_log if e["pass"] == "ner"]

    log_entry = {
        "timestamp":        datetime.now().isoformat(),
        "model_used":       OLLAMA_MODEL,
        "model_location":   "local Ollama — no data sent externally",
        "total_redactions": len(audit_log),
        "regex_redactions": len(regex_entries),
        "ner_redactions":   len(ner_entries),
        "redactions":       audit_log,
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(log_entry, fh, indent=2)

    return output_path


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

def _divider(title: str) -> None:
    width = 60
    print(f"\n{'=' * width}")
    print(title.center(width))
    print("=" * width)


def run_pipeline(raw_text: str) -> tuple | None:
    if not check_ollama_running():
        return None

    clean_text, audit_log = scrub_pii(raw_text)
    audit_path             = save_audit_log(audit_log)
    summary                = summarise_with_llama(clean_text)

    _divider("CLEANED TEXT")
    print(clean_text.strip())

    _divider("SUMMARY")
    print(summary)

    _divider("AUDIT LOG")
    print(f"Saved to : {audit_path}")
    print(f"Redactions: {len(audit_log)} total "
          f"({sum(1 for e in audit_log if e['pass']=='regex')} regex, "
          f"{sum(1 for e in audit_log if e['pass']=='ner')} NER)")

    return clean_text, summary, audit_log


# ─────────────────────────────────────────────
# EXAMPLE
# ─────────────────────────────────────────────

SAMPLE_TEXT = """
Patient: Sarah Johnson
Date of Birth: 12/04/1978
NHS Number: 485 777 3456
NI Number: AB 12 34 56 C
Address: 14 Maple Street, Colchester, CO4 3SQ
Phone: 07700 900982
Email: sarah.johnson@email.co.uk

Referral Notes:
Sarah Johnson was referred to St. James Hospital by Dr. Ahmed Patel
of the Colchester Medical Centre on 03/01/2024. The patient presents
with recurring migraines occurring 3-4 times per month and has been
prescribed sumatriptan 50mg as needed. She has a history of hypertension
managed by Amlodipine 5mg daily since 2019.

She is employed at Barclays Bank in London and reports significant
work-related stress as a potential trigger. BMI recorded at 27.4.

Follow-up appointment scheduled at St. James Hospital neurology
department within 6 weeks. GP to monitor blood pressure monthly.
Emergency contact: Michael Johnson, same address,
phone: 07700 900983.
"""

if __name__ == "__main__":
    run_pipeline(SAMPLE_TEXT)