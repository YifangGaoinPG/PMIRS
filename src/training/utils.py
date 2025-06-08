
import torch


def average_weights(model_paths):
    avg_state = None
    num_models = len(model_paths)

    for path in model_paths:
        state = torch.load(path, map_location="cpu")
        if avg_state is None:
            avg_state = state
        else:
            for k in avg_state:
                avg_state[k] += state[k]

    for k in avg_state:
        avg_state[k] /= num_models

    return avg_state
