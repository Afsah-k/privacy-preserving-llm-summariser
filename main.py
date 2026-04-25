
import re
import json
import requests     
import spacy
from datetime import datetime





# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
nlp = spacy.load("en_core_web_sm")

# address of the local Ollama server
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

REGEX_PATTERNS = [

    # NHS number format used 3 digits, space, 3 digits, space, 4 digits i.e. 485 777 3456
    ("NHS_NUMBER", r"\d{3}\s\d{3}\s\d{4}"),

    # date of birth format 15/03/1985 or 15-03-1985
    ("DATE_OF_BIRTH", r"\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{4}\b"),

    # postcode format i.e CO4 3SQ or SW1A 2AA
    ("POSTCODE", r"\b[A-Z]{1,2}\d{1,2}[A-Z\d]?\s?\d[A-Z]{2}\b"),

    # phone number format i.e 07700 900982 or +44 7700 900982
    ("PHONE_NUMBER", r"(\+?44\s?7\d{3}|\(?07\d{3}\)?)\s?\d{3}\s?\d{3}"),

    # email address
    ("EMAIL", r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b"),

    # NI number foramt i.e. AB 12 34 56 C
    ("NI_NUMBER", r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-Z]\b"),
]


 
# Detects and removes all PII from the input text
 

def scrub_pii(text):

    audit_log = []
    cleaned_text = text

    # ── PASS 1: REGEX ─────────────────────────────────────────────────────

    print("  Running regex patterns...")

    for label, pattern in REGEX_PATTERNS:
        matches = list(re.finditer(pattern, cleaned_text, re.IGNORECASE))
        for match in reversed(matches):
            original = match.group()
            replacement = f"[{label}]"

            audit_log.append({
                "pass": "regex",
                "type": label,
                "original": original,
                "replaced_with": replacement,
                "position": match.start()
            })

            cleaned_text = (
                cleaned_text[:match.start()]
                + replacement
                + cleaned_text[match.end():]
            )

    # ── PASS 2: NER ───────────────────────────────────────────────────────

    print("  Running spaCy NER...")

    doc = nlp(cleaned_text)

    #  label meanings:
    #   PERSON: people's names
    #   ORG: organisations (hospitals, companies, etc.)
    #   GPE: countries, cities, states (Geo-Political Entity)
    #   LOC: non-GPE locations (rivers, mountains, etc.)
    #   FAC: facilities (buildings, airports, bridges)

    REDACT_LABELS = {"PERSON", "ORG", "GPE", "LOC", "FAC"}

    entities = [
        ent for ent in doc.ents
        if ent.label_ in REDACT_LABELS
    ]
    entities.sort(key=lambda e: e.start_char, reverse=True)

    for ent in entities:
        original = ent.text
        replacement = f"[{ent.label_}]"

        audit_log.append({
            "pass": "ner",
            "type": ent.label_,
            "original": original,
            "replaced_with": replacement,
            "position": ent.start_char
        })

        cleaned_text = (
            cleaned_text[:ent.start_char]
            + replacement
            + cleaned_text[ent.end_char:]
        )

    return cleaned_text, audit_log



def check_ollama_running():
    """
    Tries to reach the local Ollama server.
    Returns True if it's running, False if not.
    Gives a helpful error message if it can't connect.
    """
    try:

        # check OLLAMA is running timeout=3 means "give up after 3 seconds"
        response = requests.get("http://localhost:11434", timeout=3)
        return True
    except requests.exceptions.ConnectionError:
        print("\n ERROR: Cannot connect to Ollama.")
        print(" Make sure Ollama is installed and running.")
        print(" Try running 'ollama serve' in a separate terminal.")
        print(" Download Ollama from: https://ollama.com\n")
        return False
    except requests.exceptions.Timeout:
        print("\n ERROR: Ollama connection timed out.")
        return False



# summarise with local LLAMA  

def summarise_with_llama(clean_text):


    # instruction to LLM.
    # what role the LLM is playing
    # what the input is
    # what format the output should be in
    # any rules it must follow
    prompt = f"""You are a clinical summarisation assistant working with anonymised patient records.

All personally identifiable information has already been removed from the text below and replaced with labels like [PERSON], [NHS_NUMBER], [DATE_OF_BIRTH], [ORG], etc.

Your task:
1. Write a concise clinical summary (3-5 sentences)
2. List the key medical information as bullet points
3. Note any follow-up actions mentioned

Important rules:
- Never try to guess or fill in the redacted values
- Refer to [PERSON] as "the patient"
- If you see [ORG], refer to it as "the referring organisation" or "the facility"

Anonymised clinical notes:
{clean_text}

Summary:"""

    print(f"  Sending to local Llama model ({OLLAMA_MODEL})...")
    print("  (This may take 15-60 seconds depending on your hardware)\n")

    try:
        # requests.post sends data to the server
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,

                # false stream means wait for the complete response before returning.
                # True stream would give you the text word by word as it generates
                "stream": False,

                "options": {
                    # temperature: how creative/random the output is
                    # 0.0 = very factual and consistent 
                    # 1.0 = more varied and creative
                    # For medical summaries low temperature is required
                    "temperature": 0.2,

                    # num_predict: maximum number of tokens or word piecesto generate i.e. ~500 tokens ≈ 375 words
                    "num_predict": 500,

                    # another way to control randomness is done that is only consider the top 90% most likely next words
                    "top_p": 0.9,
                }
            },
            timeout=120  # Wait up to 2 minutes for the model to respond
                        
        )

        # Check if the request succeeded
        response.raise_for_status()

        result = response.json()
        return result["response"].strip()

    except requests.exceptions.Timeout:
        return "ERROR: The model took too long to respond. Try a smaller model like 'phi3'."
    except requests.exceptions.ConnectionError:
        return "ERROR: Lost connection to Ollama during generation."
    except KeyError:
        return f"ERROR: Unexpected response format from Ollama: {response.text[:200]}"



# save audit log
def save_audit_log(audit_log, output_path="audit_log.json"):
    """
    Saves a full record of every redaction to a JSON file.
    This is your GDPR compliance trail — proof that PII was
    systematically removed before any AI processing occurred.
    """

    regex_redactions = [r for r in audit_log if r["pass"] == "regex"]
    ner_redactions = [r for r in audit_log if r["pass"] == "ner"]

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "model_used": OLLAMA_MODEL,
        "model_location": "local (Ollama) — no data sent externally",
        "total_redactions": len(audit_log),
        "regex_redactions": len(regex_redactions),
        "ner_redactions": len(ner_redactions),
        "redactions": audit_log
    }

    with open(output_path, "w") as f:
        json.dump(log_entry, f, indent=2)

    print(f"  Audit log saved → {output_path}")
    print(f"  Total redactions: {len(audit_log)} "
          f"({len(regex_redactions)} regex, {len(ner_redactions)} NER)")



def run_pipeline(raw_text):

    if not check_ollama_running():
        return

    print("\n" + "=" * 60)
    print("STEP 1: SCRUBBING PII")
    print("=" * 60)

    clean_text, audit_log = scrub_pii(raw_text)

    print("\n  ORIGINAL TEXT:")
    print("  " + "\n  ".join(raw_text.strip().split("\n")))

    print("\n  CLEANED TEXT (safe to process):")
    print("  " + "\n  ".join(clean_text.strip().split("\n")))

    print("\n" + "=" * 60)
    print("STEP 2: SAVING AUDIT LOG")
    print("=" * 60)
    save_audit_log(audit_log)

    print("\n" + "=" * 60)
    print("STEP 3: SUMMARISING WITH LOCAL LLAMA")
    print("=" * 60)
    print("  No personal data is being sent anywhere.\n")

    summary = summarise_with_llama(clean_text)

    print("  SUMMARY:")
    print("  " + "\n  ".join(summary.split("\n")))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print("  All processing done locally. No data left this machine.")

    return clean_text, summary, audit_log



if __name__ == "__main__":

    sample_text = """
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
    Emergency contact: Michael Johnson (husband), same address,
    phone: 07700 900983.
    """

    run_pipeline(sample_text)