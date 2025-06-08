
import os
import torch
import shutil
import multiprocessing as mp
from datetime import datetime
from fed_utils import average_weights

NUM_CLIENTS = 8
ROUNDS = 3
BASE_DIR = "./fl_workspace"
CHECKPOINT_TEMPLATE = os.path.join(BASE_DIR, "client_{}_model.pt")
AGGREGATED_MODEL_PATH = os.path.join(BASE_DIR, "global_model.pt")


def run_client(client_id, round_id):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(client_id % torch.cuda.device_count())
    client_ckpt = CHECKPOINT_TEMPLATE.format(client_id)
    print(f"[Client {client_id}] Training Round {round_id}")
    os.system(
        f"python main.py "
        f"--client_id {client_id} "
        f"--output {client_ckpt} "
        f"--round {round_id} "
        f"--data_split_path ./data_split/client_{client_id}.txt "
        f"--epochs 1 --distributed 0 --wandb 0 --tensorboard 0"
    )


def federated_train():
    os.makedirs(BASE_DIR, exist_ok=True)
    for round_id in range(ROUNDS):
        print(f"=== Federated Round {round_id} ===")
        processes = []
        for cid in range(NUM_CLIENTS):
            p = mp.Process(target=run_client, args=(cid, round_id))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        ckpts = [CHECKPOINT_TEMPLATE.format(cid) for cid in range(NUM_CLIENTS)]
        avg_state = average_weights(ckpts)
        torch.save(avg_state, AGGREGATED_MODEL_PATH)
        print(f"✅ Aggregated model saved to {AGGREGATED_MODEL_PATH}\n")


if __name__ == "__main__":
    federated_train()
