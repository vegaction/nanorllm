# Trajectory Viewer

This viewer is a standalone HTML debug tool for reading `Trajectory` style rollout data without changing the existing codebase.

## Files

- `trajectory_viewer.html`: self-contained viewer
- `trajectory_viewer_schema.json`: input contract
- `trajectory_viewer_sample.json`: sample data for local testing

## How To Open

1. Open [trajectory_viewer.html](/Users/sl/caitian/nanorllm/docs/trajectory_viewer.html) in a browser.
2. Click `Choose JSON File`.
3. Select a file that follows [trajectory_viewer_schema.json](/Users/sl/caitian/nanorllm/docs/trajectory_viewer_schema.json).

The page works from `file://` and does not require a local server.

## Expected Data Shape

The viewer reads one JSON object with:

- `meta`
- `trajectories`

`trajectories` is the only primary data source. Each trajectory should include:

- `task_id`
- `sample_index`
- `final_reward`
- `terminated`
- `termination_reason`
- `task.question`
- `steps`

Each step may include:

- `observation`
- `prompt_messages`
- `model_response`
- `action`
- `reward`
- `done`
- `info`
- `training`

`steps[].training` may be `null`.

## Graceful Degradation

The viewer is intentionally tolerant of partial debug dumps.

- Missing `prompt_messages` renders as `N/A`
- Missing `info` renders as `N/A`
- Missing `training` renders as `N/A`
- Empty token arrays still render with length `0`
- Invalid JSON or missing required top-level fields shows an error banner instead of a blank page

## Display Behavior

- Default selection is the first failed trajectory; if none exist, the first trajectory is selected
- List sorting is fixed: failed first, then longer trajectories, then `task_id`
- The viewer extracts `\\boxed{...}` from responses and displays:
  - `parsed_boxed_answer`
  - `has_boxed_format`
  - `last_answer_text`
- The compare panel only compares trajectories with the same `task_id`

## Sample File

Use [trajectory_viewer_sample.json](/Users/sl/caitian/nanorllm/docs/trajectory_viewer_sample.json) as the first test input. It includes:

- one successful trajectory
- one failed multi-step retry trajectory
- one response with multiple boxed answers
