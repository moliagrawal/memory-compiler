"""Paraphrase generator and robustness evaluation (Phase 7)."""

import re
import random
from typing import List, Dict, Optional, Any

SYNONYM_MAP = {
    r"\bperson\b": ["someone", "an individual", "a person", "a human"],
    r"\bopens\b": ["is opening", "opens up", "unlocks", "swings open"],
    r"\bcloses\b": ["is closing", "shuts", "pulls shut"],
    r"\bpicks up\b": ["lifts up", "takes up", "grabs", "picks"],
    r"\bputs down\b": ["places down", "sets down", "drops down"],
    r"\bsits on\b": ["takes a seat on", "sits down upon", "rests on"],
    r"\bstands up\b": ["rises up", "gets up on their feet", "stands"],
    r"\bwalks\b": ["is walking", "steps", "moves on foot", "paces"],
    r"\bruns\b": ["is running", "dashes", "sprints"],
    r"\bfridge\b": ["refrigerator", "icebox"],
    r"\brefrigerator\b": ["fridge", "cooler"],
    r"\bcup\b": ["mug", "tumbler", "drinking cup"],
    r"\bglass\b": ["drinking glass", "tumbler"],
    r"\blooking at\b": ["viewing", "staring at", "gazing at", "inspecting"],
    r"\bholds\b": ["is holding", "grasps", "grips", "carries in hand"],
    r"\bwashed\b": ["cleaned", "rinsed"],
    r"\bwashes\b": ["cleans", "is washing", "rinses"],
}


class ParaphraseAugmenter:
    """
    Generates linguistically valid paraphrases to measure grounding invariance under rephrasing.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def generate_paraphrase(self, query: str) -> str:
        """Create a paraphrased variant of the input query."""
        modified = query
        replaced = False

        # Attempt up to 2 synonym substitutions
        items = list(SYNONYM_MAP.items())
        self.rng.shuffle(items)

        for pattern, options in items:
            if re.search(pattern, modified, re.IGNORECASE):
                replacement = self.rng.choice(options)
                modified = re.sub(pattern, replacement, modified, count=1, flags=re.IGNORECASE)
                replaced = True
                if self.rng.random() > 0.5:
                    break

        if not replaced:
            # Fallback syntax variation
            if modified.lower().startswith("person "):
                modified = "A " + modified
            elif not modified.lower().startswith("a ") and not modified.lower().startswith("the "):
                modified = "The " + modified

        return modified

    def augment_dataset(self, records: List[Any]) -> List[Dict[str, Any]]:
        """Return list of (original, paraphrase) paired records."""
        paired = []
        for r in records:
            p_text = self.generate_paraphrase(r.query)
            paired.append({
                "query_id": r.query_id,
                "video_id": r.video_id,
                "original_query": r.query,
                "paraphrase_query": p_text,
                "start": r.start,
                "end": r.end
            })
        return paired
