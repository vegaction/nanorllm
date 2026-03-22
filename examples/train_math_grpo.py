import sys
import logging
import time
from pathlib import Path
import torch
from dataclasses import dataclass
from pathlib import Path

import json

# Allow running `python3 examples/run_math_episode.py` from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from nanorllm.rollout.engine import RolloutEngine
from nanorllm.agents.math_agent import MathAgent
from nanorllm.envs.math_env import MathEnv
from nanorllm.datasets.simple_math import get_simple_math_tasks
from nanorllm.trainer.trainer import run_train_epoch
from nanorllm.policy.hf_causal import HFCausalPolicy
from nanorllm.rewards.math_reward import math_reward
from nanorllm.utils.util import rollout_to_viewer_json

"""
cd /Users/sl/caitian/nanorllm
source .venv/bin/activate
python examples/train_math_grpo.py
"""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


system_prompt = """You are a careful math problem solver. Think step by step when useful, and end with a clear final answer.

Follow these rules strictly:
1) Solve the question and return exactly one final answer wrapped in \\boxed{...}.
2) In \\boxed{...}, output only the final value/expression (no words, units, punctuation, or extra spaces).
3) Never output multiple boxed answers.
"""


@dataclass
class TrainArgs:
    model_name: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    device: str = "cpu"

    clip_eps: float=0.2


    temperature: float = 1.0
    max_new_tokens: int=32
    max_steps: int = 5
    num_samples_per_task: int = 2
    max_length: int = 1024
    max_turn: int = 5

    lr: float = 1e-5
    train_batch_size: int = 3
    loss_agg_mode: str = 'seq-mean-token-mean'
    mode:str = 'token'

logger.info("Initializing training run")
args = TrainArgs()
logger.info("TrainArgs: %s", args)
engine = RolloutEngine()
agent = MathAgent(system_prompt=system_prompt)
env = MathEnv(reward_fn=math_reward, max_turn=args.max_turn)
load_start = time.perf_counter()
policy = HFCausalPolicy(model_name=args.model_name, device=args.device)
logger.info("Policy loaded in %.2fs", time.perf_counter() - load_start)
tokenizer = policy._tokenizer
optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)

def rollout_fn(task):
    return engine.run_episode(agent, env, policy, task, args)



tasks = get_simple_math_tasks()[:2]
logger.info("Selected %s tasks: %s", len(tasks), [task["task_id"] for task in tasks])

train_start = time.perf_counter()
result = run_train_epoch(
    tasks,
    rollout_fn,
    policy,
    tokenizer,
    optimizer,
    args
)
logger.info("Training run completed in %.2fs", time.perf_counter() - train_start)
logger.info("Training metrics: %s", result["metrics"])

viewer_data = rollout_to_viewer_json(result["episode_outputs"])
out_path = Path("docs/exported_trajectories.json")
out_path.write_text(json.dumps(viewer_data, ensure_ascii=False, indent=2))
logger.info("Exported viewer data to %s", out_path.resolve())