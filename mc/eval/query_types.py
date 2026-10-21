"""Query classification: rule-based + tagging into 4 fundamental grounding categories."""

import re
from enum import Enum
from typing import Dict, Any, List


class QueryType(str, Enum):
    OBJECT_ATTRIBUTE = "object_attribute"
    SINGLE_ACTION = "single_action"
    SUBJECT_OBJECT = "subject_object"
    MULTI_EVENT_ORDER = "multi_event_order"


ORDER_WORDS = {
    "then", "after", "before", "while", "next", "first", "second",
    "finally", "and then", "afterwards", "prior to", "subsequently"
}

INTERACTION_VERBS = {
    "open", "close", "take", "put", "pour", "drink", "eat", "wash", "hold",
    "cut", "clean", "cook", "pick", "drop", "wipe", "carry", "touch", "stir",
    "throw", "read", "turn", "wear", "tidy", "grasp", "press", "move"
}

INTRANSITIVE_ACTIONS = {
    "sit", "stand", "walk", "run", "smile", "laugh", "talk", "dance",
    "sleep", "wake", "wait", "look", "yawn", "cough", "nod"
}

ATTRIBUTE_WORDS = {
    "red", "blue", "green", "black", "white", "yellow", "dark", "bright",
    "wooden", "metallic", "plastic", "small", "big", "large", "closed", "opened"
}


def classify_query(query: str) -> QueryType:
    """
    Classify a video grounding query into one of 4 canonical categories:
      1. multi_event_order: queries expressing temporal succession or multiple sequential events
      2. subject_object: queries where a person interacts with an object (e.g. 'person opens fridge')
      3. object_attribute: queries focused primarily on object descriptions or attributes
      4. single_action: queries focused on simple intransitive actions
    """
    q_lower = query.lower().strip()

    # 1. Multi-event / sequential check
    for ow in ORDER_WORDS:
        pattern = rf"\b{re.escape(ow)}\b"
        if re.search(pattern, q_lower):
            return QueryType.MULTI_EVENT_ORDER

    tokens = set(re.findall(r"\b[a-z]+\b", q_lower))

    # 2. Check for object attributes
    has_attribute = bool(tokens & ATTRIBUTE_WORDS)

    # 3. Check for subject-object interactions vs single action
    has_interaction_verb = any(v in q_lower for v in INTERACTION_VERBS)
    has_intransitive = any(v in q_lower for v in INTRANSITIVE_ACTIONS)

    if has_interaction_verb:
        return QueryType.SUBJECT_OBJECT

    if has_attribute:
        return QueryType.OBJECT_ATTRIBUTE

    if has_intransitive:
        return QueryType.SINGLE_ACTION

    # Default heuristic: if contains more than 5 words or object nouns, classify as subject_object
    if len(tokens) > 4:
        return QueryType.SUBJECT_OBJECT
    return QueryType.SINGLE_ACTION


def tag_dataset_queries(records: List[Any]) -> Dict[str, str]:
    """Tag all records with their query type."""
    return {r.query_id: classify_query(r.query).value for r in records}
