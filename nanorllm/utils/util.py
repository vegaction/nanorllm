from typing import Any

import json
from pathlib import Path

def render_messages(
    messages: list[dict[str, Any]],
    add_generation_prompt: bool = False,
) -> str:
    rendered_messages: list[str] = []

    for message in messages or []:
        role = str(message.get("role", "user")).strip().upper() #先大写
        content = message.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        rendered_messages.append(f"<{role}>\n{content.strip()}")

    prompt_text = "\n\n".join(part for part in rendered_messages if part).strip()
    if add_generation_prompt:
        if prompt_text:
            prompt_text = f"{prompt_text}\n\n<ASSISTANT>\n"
        else:
            prompt_text = "<ASSISTANT>\n"
    return prompt_text



def to_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: to_jsonable(v) for k, v in value.__dict__.items()}
    return value

def rollout_to_viewer_json(episode_outputs):
    trajectories = []

    for sample_index_global, rollout in enumerate(episode_outputs):
        trajectory = rollout.trajectory
        task = rollout.task or {}
        step_views = rollout.step_views or []

        steps = []
        for i, step in enumerate(trajectory.steps):
            training = None
            if i < len(step_views):
                s = step_views[i]
                training = {
                    "advantage": to_jsonable(rollout.advantage),
                    "prompt_ids": to_jsonable(s.prompt_ids),
                    "response_ids": to_jsonable(s.response_ids),
                    "response_logprobs": to_jsonable(s.response_logprobs),
                }

            steps.append({
                "index": i,
                "observation": to_jsonable(step.observation),
                "prompt_messages": to_jsonable(step.prompt_messages),
                "model_response": to_jsonable(step.model_response),
                "action": to_jsonable(step.action),
                "reward": to_jsonable(step.reward),
                "done": to_jsonable(step.done),
                "info": to_jsonable(step.info),
                "training": training,
            })

        trajectories.append({
            "task_id": trajectory.task_id,
            "sample_index": sample_index_global,
            "final_reward": trajectory.final_reward,
            "terminated": trajectory.terminated,
            "termination_reason": trajectory.termination_reason,
            "task": {
                "question": task.get("question"),
                "answer": task.get("answer"),
            },
            "steps": steps,
        })

    return {
        "meta": {
            "env_name": "MathEnv",
            "created_at": "2026-03-19T00:00:00+08:00",
            "source": "train_math_grpo",
            "num_trajectories": len(trajectories),
        },
        "trajectories": trajectories,
    }
