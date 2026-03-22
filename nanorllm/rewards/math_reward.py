import re
from decimal import Decimal, InvalidOperation

from nanorllm.core.types import Action, RewardOutput

SPECIAL_TOKEN_PATTERN = re.compile(r"<\|[^<>]+\|>|</?s>|<eos>|<bos>|<pad>")
BOX_PATTERNS = (
    re.compile(r"\\boxed\s*\{([^{}]+)\}"),
    re.compile(r"\\box\s*\{([^{}]+)\}"),
)
FINAL_ANSWER_PATTERNS = (
    re.compile(r"final answer\s*(?:is|=|:)?\s*(.+)$", re.IGNORECASE),
    re.compile(r"answer\s*(?:is|=|:)\s*(.+)$", re.IGNORECASE),
)
NUMERIC_TOKEN_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?")




def _strip_answer_wrappers(text: str) -> str:
    text = SPECIAL_TOKEN_PATTERN.sub("", text).strip()
    text = text.strip("$")
    text = text.rstrip(".,;:!?")
    text = re.sub(r"^\((.*)\)$", r"\1", text)
    return text.strip()


def normalize_math_answer(text: str) -> str:
    cleaned = _strip_answer_wrappers(text)
    if not cleaned:
        return ""

    if NUMERIC_TOKEN_PATTERN.fullmatch(cleaned):
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            try:
                value = Decimal(numerator) / Decimal(denominator)
            except (InvalidOperation, ZeroDivisionError):
                return cleaned
        else:
            try:
                value = Decimal(cleaned)
            except InvalidOperation:
                return cleaned

        normalized = format(value.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized or "0"

    return cleaned


def extract_math_answer(text: str) -> str:
    cleaned = SPECIAL_TOKEN_PATTERN.sub("", text).strip()

    for pattern in BOX_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            return normalize_math_answer(matches[-1])

    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    for line in reversed(lines):
        for pattern in FINAL_ANSWER_PATTERNS:
            match = pattern.search(line)
            if match:
                candidate = normalize_math_answer(match.group(1))
                if candidate:
                    return candidate

        if "=" in line:
            rhs = normalize_math_answer(line.rsplit("=", maxsplit=1)[-1])
            if rhs:
                rhs_numeric_matches = NUMERIC_TOKEN_PATTERN.findall(rhs)
                if rhs_numeric_matches:
                    return normalize_math_answer(rhs_numeric_matches[-1])
                return rhs

    numeric_matches = NUMERIC_TOKEN_PATTERN.findall(cleaned)
    if numeric_matches:
        return normalize_math_answer(numeric_matches[-1])

    return normalize_math_answer(cleaned)


def ensure_boxed_math_response(text: str) -> str:
    answer = extract_math_answer(text)
    if not answer:
        return text
    return f"\\boxed{{{answer}}}"



def math_reward(task, action)-> RewardOutput:
    predicted_answer = extract_math_answer(action.value)
    is_correct = predicted_answer == task["answer"]
    return RewardOutput(
        reward=1 if is_correct else 0,
        is_correct=is_correct,
        metadata={
            "predicted_answer": predicted_answer,
            "expected_answer": task["answer"],
        },
    )
