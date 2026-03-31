from nanorllm.core.types import Action
from nanorllm.rewards.math_reward import (
    ensure_boxed_math_response,
    extract_math_answer,
    math_reward,
    normalize_math_answer,
)


def test_extract_math_answer_handles_special_tokens_without_box():
    assert extract_math_answer("96 - 57 = 39<|im_end|>") == "39"


def test_extract_math_answer_supports_box_alias():
    assert extract_math_answer(r"\box{39}") == "39"


def test_ensure_boxed_math_response_canonicalizes_plain_text_answer():
    assert ensure_boxed_math_response("96 - 57 = 39<|im_end|>") == r"\boxed{39}"


def test_normalize_math_answer_treats_integer_and_decimal_as_equal():
    assert normalize_math_answer("39.0") == "39"


def test_math_reward_accepts_correct_plain_text_answer():
    reward = math_reward({"answer": "39"}, Action("96 - 57 = 39<|im_end|>"))
    assert reward.is_correct is True
    assert reward.reward == 1


def test_math_reward_rejects_wrong_answer():
    reward = math_reward({"answer": "39"}, Action("96 - 57 = 24<|im_end|>"))
    assert reward.is_correct is False
    assert reward.reward == 0
