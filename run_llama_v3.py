# run_llama_new.py
# Reads: newcases.json, promptsv3.json
# Requires: Ollama running locally (http://localhost:11434)
# Model: llama3.1:latest
# Writes: outputs_llama_new.json, outputs_llama_new.csv

import json
import time
import csv
from typing import Dict, Any, List

import requests


OUTPUT_JSON  = "outputs_llama_v3.json"
OUTPUT_CSV   = "outputs_llama_v3.csv"
CASES_FILE   = "newcases.json"
PROMPTS_FILE = "promptv3.json"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:latest"

TEMPERATURE = 0.2
SLEEP_SECONDS_BETWEEN_CALLS = 0.2


# -----------------------------
# Template renderer
# -----------------------------
def render_template(template, problem: str, student_attempt: str) -> str:
    if isinstance(template, list):
        template = "\n".join(template)
    return (
        template.replace("{problem}", problem)
                .replace("{student_attempt}", student_attempt)
    )


# -----------------------------
# Call Ollama
# -----------------------------
def call_ollama(prompt_text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE
        }
    }

    r = requests.post(OLLAMA_URL, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "") or ""


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

        # Extract reasoning
        if "REASONING:" in reasoning_part:
            reasoning = reasoning_part.split("REASONING:", 1)[1].strip()
        else:
            reasoning = reasoning_part.strip()

        feedback = feedback_part.strip()

    else:
        # fallback if model fails format
        feedback = output.strip()
        reasoning = ""

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
# Main
# -----------------------------
def main():
    # Load dataset
    with open(CASES_FILE, "r", encoding="utf-8") as f:
        cases: List[Dict[str, Any]] = json.load(f)

    # Load prompts
    with open(PROMPTS_FILE, "r", encoding="utf-8") as f:
        prompt_data: Dict[str, Any] = json.load(f)

    prompts = prompt_data["prompts"]

    required_keys = ["zero_shot", "standard_cot", "pcot", "taglish_pcot"]
    missing = [k for k in required_keys if k not in prompts]
    if missing:
        raise RuntimeError(f"promptsv3.json is missing required keys: {missing}")

    outputs: List[Dict[str, Any]] = []

    for c in cases:
        group            = c.get("group")
        group_label      = c.get("group_label", "")
        case_id          = c.get("case_id")
        misconception_id = c.get("misconception_id", "")
        problem          = c.get("problem", "")
        student_attempt  = c.get("student_attempt", "")

        if group is None or not case_id or not problem:
            raise RuntimeError(f"Invalid case entry: {c}")

        # -------------------
        # Zero-shot
        # -------------------
        zero_prompt = render_template(
            prompts["zero_shot"]["template"], problem, student_attempt
        )
        zero_out = call_ollama(zero_prompt)
        time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

        # -------------------
        # Standard CoT
        # -------------------
        cot_prompt = render_template(
            prompts["standard_cot"]["template"], problem, student_attempt
        )
        cot_out = call_ollama(cot_prompt)
        time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

        # -------------------
        # P-CoT
        # -------------------
        pcot_prompt = render_template(
            prompts["pcot"]["template"], problem, student_attempt
        )
        pcot_raw = call_ollama(pcot_prompt)
        pcot_split = split_reasoning_feedback(pcot_raw)
        time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

        # -------------------
        # Taglish P-CoT
        # -------------------
        tag_prompt = render_template(
            prompts["taglish_pcot"]["template"], problem, student_attempt
        )
        tag_raw = call_ollama(tag_prompt)
        tag_split = split_reasoning_feedback(tag_raw)
        time.sleep(SLEEP_SECONDS_BETWEEN_CALLS)

        # -------------------
        # Save outputs
        # -------------------
        outputs.extend([
            {
                "group": group,
                "group_label": group_label,
                "case_id": case_id,
                "misconception_id": misconception_id,
                "model": OLLAMA_MODEL,
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
                "model": OLLAMA_MODEL,
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
                "model": OLLAMA_MODEL,
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
                "model": OLLAMA_MODEL,
                "prompt_type": "taglish_pcot",
                "problem": problem,
                "student_attempt": student_attempt,
                "reasoning_output": tag_split["reasoning_output"],
                "feedback_output": tag_split["feedback_output"]
            }
        ])

        print(f"Llama finished: {case_id} (Group {group} — {group_label})")

    # Save JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(outputs, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUTPUT_JSON} with {len(outputs)} rows.")

    # Save CSV
    save_csv(outputs, OUTPUT_CSV)
    print(f"Saved {OUTPUT_CSV} with {len(outputs)} rows.")


if __name__ == "__main__":
    main()