# llm_judge.py
# Reads:  outputs_gemini_v3.json + outputs_llama_v3.json
# Requires: pip install anthropic python-dotenv
# Env: ANTHROPIC_API_KEY=your_key_here
# Model: claude-sonnet-4-5
# Writes: outputs_llm_judge.json + outputs_llm_judge.csv

import os
import json
import time
import csv
import re
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from dotenv import load_dotenv
import anthropic

# -----------------------------
# Config
# -----------------------------
OUTPUT_JSON   = "outputs_llm_judge.json"
OUTPUT_CSV    = "outputs_llm_judge.csv"
GEMINI_FILE   = "outputs_gemini_v3.json"
LLAMA_FILE    = "outputs_llama_v3.json"

MODEL         = "claude-sonnet-4-5"
MAX_TOKENS    = 200
TEMPERATURE   = 0.0
SLEEP_BETWEEN = 1.0
MAX_RETRIES   = 3
RETRY_WAIT    = 10

# -----------------------------
# Custom exceptions
# -----------------------------
class CreditError(Exception):
    pass

class QuotaError(Exception):
    pass

# -----------------------------
# Judge prompt
# -----------------------------
JUDGE_PROMPT = """You are an expert educational evaluator assessing the quality of AI-generated feedback on student algebra errors.

You will be given:
1. A math problem
2. A student's incorrect answer
3. An AI-generated feedback response

Rate the feedback on EXACTLY these 4 dimensions using a 4-point scale:

DIMENSION 1 — Diagnostic Accuracy
Definition: The extent to which the feedback correctly identifies, addresses, and explains the student's actual mathematical error or misconception.
4 = Excellent: The feedback accurately identifies the student's error or misconception and explains it correctly without introducing mathematical confusion.
3 = Good: The feedback is mostly accurate and addresses the main error, though the explanation may be slightly incomplete or lack precision.
2 = Fair: The feedback shows partial understanding of the error but is vague, incomplete, or somewhat misleading in explaining the mistake.
1 = Poor: The feedback misidentifies the error, explains it incorrectly, or gives mathematically inaccurate guidance.

DIMENSION 2 — Pedagogical Scaffolding
Definition: The extent to which the feedback provides supportive, step-by-step, and learner-centered guidance that helps the student understand and correct the error.
4 = Excellent: The feedback is highly supportive, well-structured, and clearly guides the student through understanding the mistake and how to improve.
3 = Good: The feedback is generally helpful and supportive, with some guidance toward correction, though the explanation may not be fully developed.
2 = Fair: The feedback gives limited support and only partially helps the student understand or correct the mistake.
1 = Poor: The feedback is abrupt, overly brief, confusing, or does not meaningfully help the student learn from the error.

DIMENSION 3 — Linguistic Naturalness
Definition: The extent to which the feedback sounds clear, natural, learner-appropriate, and easy to understand, in a way that resembles effective classroom communication.
4 = Excellent: The feedback is very clear, natural, easy to follow, and appropriate for the learner's level.
3 = Good: The feedback is mostly clear and understandable, though some wording may sound slightly awkward, formal, or less smooth.
2 = Fair: The feedback is somewhat unnatural, unclear, or difficult to follow in parts.
1 = Poor: The feedback sounds unnatural, confusing, overly technical, or inappropriate for the learner's level.

DIMENSION 4 — Actionability
Definition: The extent to which the feedback gives the student a clear sense of what to do next in order to correct the error or improve their solution.
4 = Excellent: The feedback gives clear, practical, and specific guidance on how the student can correct the error or proceed.
3 = Good: The feedback gives useful direction, though it may be somewhat general or not fully specific.
2 = Fair: The feedback suggests improvement only vaguely and does not clearly show the student what steps to take next.
1 = Poor: The feedback provides little to no actionable guidance for correcting the error.

IMPORTANT RULES:
- Rate ONLY the feedback quality, not the math problem itself
- Do NOT be lenient — use the full 1-4 range
- For feedback in Taglish (Filipino-English mix), evaluate Linguistic Naturalness based on how natural the Taglish sounds for a Filipino learner
- Return ONLY a JSON object with exactly these keys: diagnostic_accuracy, pedagogical_scaffolding, linguistic_naturalness, actionability
- Each value must be an integer 1, 2, 3, or 4
- No explanation, no other text — ONLY the JSON object

Example valid response:
{{"diagnostic_accuracy": 3, "pedagogical_scaffolding": 4, "linguistic_naturalness": 3, "actionability": 2}}

Now rate the following:

PROBLEM:
{problem}

STUDENT'S ANSWER:
{student_attempt}

AI-GENERATED FEEDBACK:
{feedback}
"""

# -----------------------------
# Helpers
# -----------------------------
def load_json(filepath: str) -> Any:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_existing(filepath: str) -> List[Dict[str, Any]]:
    if not Path(filepath).exists():
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(outputs: List[Dict[str, Any]], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)

def save_csv(outputs: List[Dict[str, Any]], filepath: str) -> None:
    fieldnames = [
        "group", "group_label", "case_id", "misconception_id",
        "model", "prompt_type", "llm_judge_model",
        "diagnostic_accuracy", "pedagogical_scaffolding",
        "linguistic_naturalness", "actionability"
    ]
    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)

def save_progress(outputs: List[Dict[str, Any]]) -> None:
    save_json(outputs, OUTPUT_JSON)
    save_csv(outputs, OUTPUT_CSV)

def build_done_keys(outputs: List[Dict[str, Any]]) -> Set[Tuple[str, str, str]]:
    return {(r["case_id"], r["prompt_type"], r["model"]) for r in outputs}

def is_credit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return "credit balance is too low" in msg or "upgrade or purchase credits" in msg

def is_quota_error(e: Exception) -> bool:
    msg = str(e).upper()
    return any(x in msg for x in ["429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA"])

# -----------------------------
# Claude API call
# -----------------------------
def call_claude(client: anthropic.Anthropic, prompt: str) -> Dict[str, int]:
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}]
            )
            text = response.content[0].text.strip()
            text = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
            scores = json.loads(text)
            required = ["diagnostic_accuracy", "pedagogical_scaffolding",
                        "linguistic_naturalness", "actionability"]
            for key in required:
                if key not in scores:
                    raise ValueError(f"Missing key: {key}")
                if scores[key] not in [1, 2, 3, 4]:
                    raise ValueError(f"Invalid score for {key}: {scores[key]}")
            return scores

        except Exception as e:
            # Stop immediately on credit errors — no point retrying
            if is_credit_error(e):
                raise CreditError(str(e)) from e
            # Stop immediately on quota/rate limit
            if is_quota_error(e):
                raise QuotaError(str(e)) from e

            last_error = e
            print(f"    Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

    raise RuntimeError(f"Claude call failed after {MAX_RETRIES} attempts: {last_error}")

# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ANTHROPIC_API_KEY in your .env file.\n"
            "Add this line: ANTHROPIC_API_KEY=your_key_here"
        )

    client = anthropic.Anthropic(api_key=api_key)

    gemini      = load_json(GEMINI_FILE)
    llama       = load_json(LLAMA_FILE)
    all_outputs = gemini + llama
    print(f"Loaded {len(all_outputs)} outputs to evaluate.")

    results   = load_existing(OUTPUT_JSON)
    done_keys = build_done_keys(results)
    print(f"Already done: {len(done_keys)} | Remaining: {len(all_outputs) - len(done_keys)}")

    total = len(all_outputs)
    try:
        for idx, entry in enumerate(all_outputs, 1):
            key = (entry["case_id"], entry["prompt_type"], entry["model"])

            if key in done_keys:
                print(f"[{idx}/{total}] Skipping {entry['case_id']} | {entry['prompt_type']} | {entry['model']}")
                continue

            print(f"[{idx}/{total}] Rating {entry['case_id']} | {entry['prompt_type']} | {entry['model']} ...", end=" ", flush=True)

            prompt = JUDGE_PROMPT.format(
                problem=entry["problem"],
                student_attempt=entry["student_attempt"],
                feedback=entry["feedback_output"]
            )

            try:
                scores = call_claude(client, prompt)
                result = {
                    "group":                   entry["group"],
                    "group_label":             entry["group_label"],
                    "case_id":                 entry["case_id"],
                    "misconception_id":        entry["misconception_id"],
                    "model":                   entry["model"],
                    "prompt_type":             entry["prompt_type"],
                    "llm_judge_model":         MODEL,
                    "diagnostic_accuracy":     scores["diagnostic_accuracy"],
                    "pedagogical_scaffolding": scores["pedagogical_scaffolding"],
                    "linguistic_naturalness":  scores["linguistic_naturalness"],
                    "actionability":           scores["actionability"],
                }
                results.append(result)
                done_keys.add(key)
                save_progress(results)
                print(f"DA={scores['diagnostic_accuracy']} PS={scores['pedagogical_scaffolding']} LN={scores['linguistic_naturalness']} AC={scores['actionability']}")
                time.sleep(SLEEP_BETWEEN)

            except CreditError:
                save_progress(results)
                print(f"\n\nSTOPPED: Your Anthropic API credit balance is too low.")
                print("Please go to console.anthropic.com → Plans & Billing to add credits.")
                print(f"Progress saved. {len(results)} items done so far.")
                print("Run the script again after adding credits — it will resume automatically.")
                return

            except QuotaError:
                save_progress(results)
                print(f"\n\nSTOPPED: Rate limit or quota reached.")
                print(f"Progress saved. {len(results)} items done so far.")
                print("Wait a few minutes then run the script again — it will resume.")
                return

            except Exception as e:
                print(f"\n  ERROR: {e} — skipping this entry.")
                continue

    except KeyboardInterrupt:
        save_progress(results)
        print(f"\n\nStopped by user. Progress saved. {len(results)} items done.")
        return

    save_progress(results)
    print(f"\nDone! {len(results)} ratings saved.")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  CSV:  {OUTPUT_CSV}")

if __name__ == "__main__":
    main()