"""Small fixed math dataset for stage-3 batch rollout."""

from __future__ import annotations


SIMPLE_MATH_20: list[dict[str, str]] = [
    {"task_id": "gsm8k-001", "question": "17 + 28 = ?", "answer": "45"},
    {"task_id": "gsm8k-002", "question": "96 - 57 = ?", "answer": "39"},
    {"task_id": "gsm8k-003", "question": "14 * 6 = ?", "answer": "84"},
    {"task_id": "gsm8k-004", "question": "144 / 12 = ?", "answer": "12"},
    {"task_id": "gsm8k-005", "question": "(8 + 4) * 3 - 10 = ?", "answer": "26"},
    {"task_id": "gsm8k-006", "question": "5^2 + 3^2 = ?", "answer": "34"},
    {"task_id": "gsm8k-007", "question": "Solve for x: 3x + 7 = 25", "answer": "6"},
    {"task_id": "gsm8k-008", "question": "Solve for x: 4x - 9 = 35", "answer": "11"},
    {"task_id": "gsm8k-009", "question": "If y / 5 = 9, y = ?", "answer": "45"},
    {"task_id": "gsm8k-010", "question": "What is the sum of integers from 1 to 10?", "answer": "55"},
    {"task_id": "gsm8k-011", "question": "Average of 10, 14, and 18 = ?", "answer": "14"},
    {"task_id": "gsm8k-012", "question": "Rectangle area with length 9 and width 7 = ?", "answer": "63"},
    {"task_id": "gsm8k-013", "question": "Perimeter of a square with side 11 = ?", "answer": "44"},
    {"task_id": "gsm8k-014", "question": "30% of 250 = ?", "answer": "75"},
    {"task_id": "gsm8k-015", "question": "Increase 80 by 15% = ?", "answer": "92"},
    {"task_id": "gsm8k-016", "question": "2 * (3 + 5)^2 = ?", "answer": "128"},
    {"task_id": "gsm8k-017", "question": "Combination C(7,2) = ?", "answer": "21"},
    {"task_id": "gsm8k-018", "question": "2^5 - 3^3 = ?", "answer": "5"},
    {"task_id": "gsm8k-019", "question": "If x + y = 13 and x - y = 5, x = ?", "answer": "9"},
    {"task_id": "gsm8k-020", "question": "At 60 km/h for 2.5 hours, distance = ? (km)", "answer": "150"},
]


def get_simple_math_tasks() -> list[dict[str, str]]:
    """Return a shallow-copied task list for safe reuse by callers."""
    return [task.copy() for task in SIMPLE_MATH_20]

