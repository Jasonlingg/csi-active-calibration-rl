"""A smallest-margin baseline; this is not an optimal query oracle."""
from harbor_client import fetch_state, query

state = fetch_state()
while not state["terminated"]:
    observation = state["observation"]
    candidates = []
    for index, valid in enumerate(observation["action_mask"]):
        if valid:
            ordered = sorted(observation["scores"][index])
            candidates.append((ordered[-1] - ordered[-2], index))
    _, selected = min(candidates)
    state = query(selected)
print("Completed", state["queries_used"], "requests; reward", state["reward"])
