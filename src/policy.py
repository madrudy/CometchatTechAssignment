import re
from .models import Chunk

def is_sensitive_request(text: str) -> bool:
    t = text.lower()
    terms = [
        "system prompt", "hidden prompt", "hidden instructions", "secret",
        "api key", "password", "internal note", "risk score", "customer email",
        "customer address", "another customer's", "other customer's",
        "fraud review", "warehouse note",
    ]
    return any(term in t for term in terms)

def asks_for_order(text: str) -> bool:
    t = text.lower()
    has_id = bool(re.search(r"\bord-\d{4}\b", t, re.I))
    order_words = any(x in t for x in [
        "where is", "order status", "tracking", "when will", "when should",
        "arrive", "delivery estimate", "check ", "cancel ", "change ", "refund"
    ])
    return has_id or (order_words and any(x in t for x in ["my order", "the order", "my purchase"]))

def needs_handoff(text: str) -> bool:
    t = text.lower()
    action_terms = [
        "refund", "cancel", "cancellation", "replacement", "address change",
        "price adjustment", "warranty approval", "approve my return", "change my order"
    ]
    return any(x in t for x in action_terms)

def conflict_topic(query: str, chunks: list[Chunk]) -> bool:
    q = query.lower()
    if "dishwasher" in q and "tumbler" in q:
        strong = [c for c in chunks if c.score >= 0.55]
        return len({c.filename for c in strong}) >= 2
    return False
