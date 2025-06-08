# --- federated.py ---
import multiprocessing as mp
import os
import torch
from main import main  # ✅ 导入原始训练逻辑

NUM_CLIENTS = 8
ROUNDS = 3
BASE_DIR = "checkpoints"
CHECKPOINT_TEMPLATE = os.path.join(BASE_DIR, "client_{}_latest.pt")
AGGREGATED_MODEL_PATH = os.path.join(BASE_DIR, "aggregated_model.pt")

def run_client(cid, round_id):
    print(f"--> Client {cid} training for round {round_id}")
    
    # 你可以根据需要在这里配置 sys.argv 参数或用 argsparser 的方式传参给 main
    # 如果 main() 从 argparse 读取参数，则你需要 mock sys.argv
    import sys
    sys.argv = [
        "main.py",
        "--name", f"client_{cid}_round_{round_id}",
        "--epochs", "1",
        "--resume", CHECKPOINT_TEMPLATE.format(cid) if round_id > 0 else "",
        "--save", CHECKPOINT_TEMPLATE.format(cid),
        "--batch-size", "64",
        "--local_rank", "0",  # 非分布式模拟
        "--dist-url", "env://",
        "--dist-backend", "nccl"
    ]
    
    main()

def average_weights(ckpt_paths):
    state_dicts = [torch.load(path, map_location='cpu') for path in ckpt_paths]
    avg_state = {}
    for key in state_dicts[0].keys():
        avg_state[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
    return avg_state

def federated_train():
    os.makedirs(BASE_DIR, exist_ok=True)
    for round_id in range(ROUNDS):
        print(f"\n=== Federated Round {round_id} ===")
        processes = []
        for cid in range(NUM_CLIENTS):
            p = mp.Process(target=run_client, args=(cid, round_id))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        # Aggregate model checkpoints
        ckpts = [CHECKPOINT_TEMPLATE.format(cid) for cid in range(NUM_CLIENTS)]
        avg_state = average_weights(ckpts)
        torch.save(avg_state, AGGREGATED_MODEL_PATH)
        print(f"✅ Aggregated model saved to {AGGREGATED_MODEL_PATH}\n")

if __name__ == "__main__":
    federated_train()
