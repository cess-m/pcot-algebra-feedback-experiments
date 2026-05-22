# run_gemini_v3_resume.py
# Reads: newcases.json, promptv3.json, outputs_gemini_v3.json (if it exists)
# Requires: pip install -U google-genai python-dotenv
# Env: GEMINI_API_KEY=your_key_here
# Model: gemini-2.5-flash
# Writes: outputs_gemini_v3.json, outputs_gemini_v3.csv

import os
import json
import time
import csv
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple

from dotenv import load_dotenv
from google import genai


OUTPUT_JSON = "outputs_gemini_v3.json"
OUTPUT_CSV = "outputs_gemini_v3.csv"
CASES_FILE = "newcases.json"
PROMPTS_FILE = "promptv3.json"

GEMINI_MODEL = "gemini-2.5-flash"

TEMPERATURE = 0.2
SLEEP_SECONDS_BETWEEN_CALLS = 1

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 8


class QuotaExceededError(Exception):
    pass


# -----------------------------
# Template renderer
# -----------------------------
def render_template(template: Any, problem: str, student_attempt: str) -> str:
    if isinstance(template, list):
        template = "\n".join(template)

    return (
        str(template)
        .replace("{problem}", problem)
        .replace("{student_attempt}", student_attempt)
    )


# -----------------------------
# File helpers
# -----------------------------
def ensure_file_exists(filepath: str) -> None:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")


def load_json(filepath: str) -> Any:
    ensure_file_exists(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing_outputs(filepath: str) -> List[Dict[str, Any]]:
    if not Path(filepath).exists():
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise RuntimeError(f"{filepath} exists but does not contain a JSON list.")

    return data


# -----------------------------
# Save helpers
# -----------------------------
def save_json(outputs: List[Dict[str, Any]], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


def save_csv(outputs: List[Dict[str, Any]], filepath: str) -> None:
    fieldnames = [
        "group", "group_label", "case_id", "misconception_id",
        "model", "prompt_type", "problem", "student_attempt",
        "reasoning_output", "feedback_output"
    ]

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(outputs)


def save_progress(outputs: List[Dict[str, Any]]) -> None:
    save_json(outputs, OUTPUT_JSON)
    save_csv(outputs, OUTPUT_CSV)


# -----------------------------
# Existing completed pairs
# -----------------------------
def build_completed_set(outputs: List[Dict[str, Any]]) -> Set[Tuple[str, str]]:
    completed = set()

    for row in outputs:
        case_id = row.get("case_id")
        prompt_type = row.get("prompt_type")

        if case_id and prompt_type:
            completed.add((case_id, prompt_type))

    return completed


# -----------------------------
# Detect quota errors
# -----------------------------
def is_quota_error(error: Exception) -> bool:
    msg = str(error).upper()
    signals = [
        "429",
        "RESOURCE_EXHAUSTED",
        "QUOTA EXCEEDED",
        "EXCEEDED YOUR CURRENT QUOTA",
        "GENERATE_CONTENT_FREE_TIER_REQUESTS",
        "RATE LIMIT"
    ]
    return any(signal in msg for signal in signals)


# -----------------------------
# Gemini call
# -----------------------------
def call_gemini(client: genai.Client, prompt_text: str) -> str:
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt_text,
                config={
                    "temperature": TEMPERATURE
                }
            )

            text = getattr(response, "text", None)
            if text:
                return text.strip()

            return ""

        except Exception as e:
            last_error = e

            if is_quota_error(e):
                raise QuotaExceededError(str(e)) from e

            print(f"Gemini call failed on attempt {attempt}/{MAX_RETRIES}: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


# -----------------------------
# Output parsing
# -----------------------------
def split_reasoning_feedback(output: str) -> Dict[str, str]:
    reasoning = ""
    feedback = output.strip()

    if "FEEDBACK:" in output:
        parts = output.split("FEEDBACK:", 1)
        reasoning_part = parts[0]
        feedback_part = parts[1]

        if "REASONING:" in reasoning_part:
            reasoning = reasoning_part.split("REASONING:", 1)[1].strip()
        else:
            reasoning = reasoning_part.strip()

        feedback = feedback_part.strip()

    return {
        "reasoning_output": reasoning,
        "feedback_output": feedback
    }


def run_prompt(
    client: genai.Client,
    prompt_template: Any,
    problem: str,
    student_attempt: str
) -> str:
    prompt_text = render_template(prompt_template, problem, student_attempt)
    result = call_gemini(client, prompt_text)
    time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)
    return result


def build_output_row(
    group: Any,
    group_label: str,
    case_id: str,
    misconception_id: str,
    prompt_type: str,
    problem: str,
    student_attempt: str,
    reasoning_output: str,
    feedback_output: str
) -> Dict[str, Any]:
    return {
        "group": group,
        "group_label": group_label,
        "case_id": case_id,
        "misconception_id": misconception_id,
        "model": GEMINI_MODEL,
        "prompt_type": prompt_type,
        "problem": problem,
        "student_attempt": student_attempt,
        "reasoning_output": reasoning_output,
        "feedback_output": feedback_output
    }


def append_and_save(
    outputs: List[Dict[str, Any]],
    completed: Set[Tuple[str, str]],
    row: Dict[str, Any]
) -> None:
    outputs.append(row)
    completed.add((row["case_id"], row["prompt_type"]))
    save_progress(outputs)


# -----------------------------
# Main
# -----------------------------
def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing GEMINI_API_KEY in your .env file. "
            "Example: GEMINI_API_KEY=your_actual_key_here"
        )

    client = genai.Client(api_key=api_key)

    cases: List[Dict[str, Any]] = load_json(CASES_FILE)
    prompt_data: Dict[str, Any] = load_json(PROMPTS_FILE)

    if "prompts" not in prompt_data:
        raise RuntimeError(f"{PROMPTS_FILE} must contain a top-level 'prompts' key.")

    prompts = prompt_data["prompts"]

    required_keys = ["zero_shot", "standard_cot", "pcot", "taglish_pcot"]
    missing = [k for k in required_keys if k not in prompts]
    if missing:
        raise RuntimeError(f"{PROMPTS_FILE} is missing required keys: {missing}")

    for key in required_keys:
        if "template" not in prompts[key]:
            raise RuntimeError(f"{PROMPTS_FILE}: prompt '{key}' is missing a 'template' field.")

    outputs: List[Dict[str, Any]] = load_existing_outputs(OUTPUT_JSON)
    completed = build_completed_set(outputs)

    print(f"Loaded {len(outputs)} existing rows from {OUTPUT_JSON}")
    print(f"Found {len(completed)} completed (case_id, prompt_type) pairs")
    print(f"Resuming with model: {GEMINI_MODEL}")

    prompt_plan = [
        ("zero_shot", "zero_shot"),
        ("standard_cot", "standard_cot"),
        ("pcot", "pcot"),
        ("taglish_pcot", "taglish_pcot"),
    ]

    try:
        total_cases = len(cases)

        for idx, c in enumerate(cases, start=1):
            group = c.get("group")
            group_label = c.get("group_label", "")
            case_id = c.get("case_id")
            misconception_id = c.get("misconception_id", "")
            problem = c.get("problem", "")
            student_attempt = c.get("student_attempt", "")

            if group is None or not case_id or not problem:
                print(f"Skipping invalid case: {c}")
                continue

            print(f"\n[{idx}/{total_cases}] Processing {case_id} (Group {group} — {group_label})")

            for prompt_type, prompt_key in prompt_plan:
                if (case_id, prompt_type) in completed:
                    print(f"Skipping already completed: {case_id} - {prompt_type}")
                    continue

                try:
                    raw_output = run_prompt(
                        client,
                        prompts[prompt_key]["template"],
                        problem,
                        student_attempt
                    )

                    if prompt_type in {"pcot", "taglish_pcot"}:
                        split = split_reasoning_feedback(raw_output)
                        row = build_output_row(
                            group=group,
                            group_label=group_label,
                            case_id=case_id,
                            misconception_id=misconception_id,
                            prompt_type=prompt_type,
                            problem=problem,
                            student_attempt=student_attempt,
                            reasoning_output=split["reasoning_output"],
                            feedback_output=split["feedback_output"]
                        )
                    else:
                        row = build_output_row(
                            group=group,
                            group_label=group_label,
                            case_id=case_id,
                            misconception_id=misconception_id,
                            prompt_type=prompt_type,
                            problem=problem,
                            student_attempt=student_attempt,
                            reasoning_output="",
                            feedback_output=raw_output.strip()
                        )

                    append_and_save(outputs, completed, row)
                    print(f"Saved: {case_id} - {prompt_type}")

                except QuotaExceededError:
                    save_progress(outputs)
                    print(f"\nQuota/rate limit reached during {case_id} - {prompt_type}")
                    print("Progress saved. Stopping run now.")
                    return

                except Exception as e:
                    print(f"Error in {case_id} - {prompt_type}: {e}")

            print(f"Finished processing case: {case_id}")

    finally:
        save_progress(outputs)

    print(f"\nSaved {OUTPUT_JSON} with {len(outputs)} rows.")
    print(f"Saved {OUTPUT_CSV} with {len(outputs)} rows.")
    print("Gemini resume run complete.")


if __name__ == "__main__":
    main()