# run_gemini_v3.py
# Reads: newcases.json, promptv3.json
# Requires: pip install -U google-genai python-dotenv
# Env: GEMINI_API_KEY=your_key_here
# Model: gemini-2.5-flash
# Writes: outputs_gemini_v3.json, outputs_gemini_v3.csv

import os
import json
import time
import csv
from pathlib import Path
from typing import Dict, Any, List

from dotenv import load_dotenv
from google import genai


OUTPUT_JSON = "outputs_gemini_v3.json"
OUTPUT_CSV = "outputs_gemini_v3.csv"
CASES_FILE = "newcases.json"
PROMPTS_FILE = "promptv3.json"

GEMINI_MODEL = "gemini-2.5-flash"

TEMPERATURE = 0.2
SLEEP_SECONDS_BETWEEN_CALLS = 0.5

MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 5


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
# File checks
# -----------------------------
def ensure_file_exists(filepath: str) -> None:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"File not found: {filepath}")


# -----------------------------
# Load JSON safely
# -----------------------------
def load_json(filepath: str) -> Any:
    ensure_file_exists(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Call Gemini with retry
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
            print(f"Gemini call failed on attempt {attempt}/{MAX_RETRIES}: {e}")

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    raise RuntimeError(f"Gemini call failed after {MAX_RETRIES} attempts: {last_error}")


# -----------------------------
# Split REASONING / FEEDBACK
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


# -----------------------------
# Save CSV
# -----------------------------
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


# -----------------------------
# Save JSON
# -----------------------------
def save_json(outputs: List[Dict[str, Any]], filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)


# -----------------------------
# Run one prompt type
# -----------------------------
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

    outputs: List[Dict[str, Any]] = []

    total_cases = len(cases)
    print(f"Starting Gemini run for {total_cases} cases using model: {GEMINI_MODEL}")

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

        try:
            # Zero-shot
            zero_out = run_prompt(
                client,
                prompts["zero_shot"]["template"],
                problem,
                student_attempt
            )

            # Standard CoT
            cot_out = run_prompt(
                client,
                prompts["standard_cot"]["template"],
                problem,
                student_attempt
            )

            # P-CoT
            pcot_raw = run_prompt(
                client,
                prompts["pcot"]["template"],
                problem,
                student_attempt
            )
            pcot_split = split_reasoning_feedback(pcot_raw)

            # Taglish P-CoT
            tag_raw = run_prompt(
                client,
                prompts["taglish_pcot"]["template"],
                problem,
                student_attempt
            )
            tag_split = split_reasoning_feedback(tag_raw)

            outputs.extend([
                {
                    "group": group,
                    "group_label": group_label,
                    "case_id": case_id,
                    "misconception_id": misconception_id,
                    "model": GEMINI_MODEL,
                    "prompt_type": "zero_shot",
                    "problem": problem,
                    "student_attempt": student_attempt,
                    "reasoning_output": "",
                    "feedback_output": zero_out.strip()
                },
                {
                    "group": group,
                    "group_label": group_label,
                    "case_id": case_id,
                    "misconception_id": misconception_id,
                    "model": GEMINI_MODEL,
                    "prompt_type": "standard_cot",
                    "problem": problem,
                    "student_attempt": student_attempt,
                    "reasoning_output": "",
                    "feedback_output": cot_out.strip()
                },
                {
                    "group": group,
                    "group_label": group_label,
                    "case_id": case_id,
                    "misconception_id": misconception_id,
                    "model": GEMINI_MODEL,
                    "prompt_type": "pcot",
                    "problem": problem,
                    "student_attempt": student_attempt,
                    "reasoning_output": pcot_split["reasoning_output"],
                    "feedback_output": pcot_split["feedback_output"]
                },
                {
                    "group": group,
                    "group_label": group_label,
                    "case_id": case_id,
                    "misconception_id": misconception_id,
                    "model": GEMINI_MODEL,
                    "prompt_type": "taglish_pcot",
                    "problem": problem,
                    "student_attempt": student_attempt,
                    "reasoning_output": tag_split["reasoning_output"],
                    "feedback_output": tag_split["feedback_output"]
                }
            ])

            # Save progress after each case
            save_json(outputs, OUTPUT_JSON)
            save_csv(outputs, OUTPUT_CSV)

            print(f"Finished: {case_id}")

        except Exception as e:
            print(f"Error while processing {case_id}: {e}")
            continue

    save_json(outputs, OUTPUT_JSON)
    print(f"\nSaved {OUTPUT_JSON} with {len(outputs)} rows.")

    save_csv(outputs, OUTPUT_CSV)
    print(f"Saved {OUTPUT_CSV} with {len(outputs)} rows.")

    print("Gemini run complete.")


if __name__ == "__main__":
    main()