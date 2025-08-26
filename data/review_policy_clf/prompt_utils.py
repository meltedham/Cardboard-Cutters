# prompt_utils.py
"""
Prompt engineering helpers for classifying reviews into policy categories.
Categories (customise as needed):
- advertisement: self-promotion, discount codes, URLs encouraging purchase, "DM me", "follow my page"
- irrelevant: off-topic, questions about unrelated topics, generic emoji-only posts
- rant_without_visit: rants/complaints that clearly admit they didn't visit (e.g., "never been there but...")

The prompt template instructs an LLM to output ONLY one of the allowed labels.
"""

from dataclasses import dataclass
from typing import List, Dict

POLICY_LABELS = ["advertisement", "irrelevant", "rant_without_visit", "normal_review"]

INSTRUCTIONS = """\
You are a strict content classifier for restaurant reviews.
Classify the user's review into exactly ONE of these categories:
- advertisement: self-promotion, marketing language, coupons/discount codes, links or handles for promotion.
- irrelevant: off-topic content not describing the dining experience, emoji-only or random chatter.
- rant_without_visit: complaints or opinions that explicitly indicate the author did NOT visit.
- normal_review: a genuine dining-related review (positive, negative, or mixed) that seems to follow a visit.

Output ONLY the category label (no punctuation, no explanation).
Allowed labels: {allowed}.
"""

EXAMPLES = [
    {"text": "Check out my page @dealhunters for 50% off coupons! Limited time!", "label": "advertisement"},
    {"text": "Anyone knows when the MRT opens tomorrow?", "label": "irrelevant"},
    {"text": "Never been to this place but I hate their vibe already.", "label": "rant_without_visit"},
    {"text": "Food was tasty and the staff were attentive. Would return.", "label": "normal_review"},
]

def build_prompt(review_text: str, add_few_shots: bool = True) -> str:
    head = INSTRUCTIONS.format(allowed=", ".join(POLICY_LABELS)).strip()
    shots = ""
    if add_few_shots and EXAMPLES:
        lines = []
        for ex in EXAMPLES:
            lines.append(f'Review: "{ex["text"]}"\nLabel: {ex["label"]}')
        shots = "\n\n".join(lines)
    user = f'Review: "{review_text}"\nLabel:'
    return "\n\n".join([head, shots, user]) if shots else "\n\n".join([head, user])

def parse_label(raw_output: str) -> str:
    raw = (raw_output or "").strip().lower()
    raw = raw.splitlines()[0].strip().strip(" .,:;[]{}()\"'")
    for lbl in POLICY_LABELS:
        if raw.startswith(lbl):
            return lbl
    return "normal_review"
