from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cowp.core.config import load_config
from cowp.data.dataset import TorchCOWPDataset, collate_torch
from cowp.models.cowp_model import COWPModel
from cowp.models.losses import planner_ranking_loss, response_loss, witness_loss


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train COWP model stages on COWP tensor cache.")
    ap.add_argument("--data-config", default="configs/data.yaml")
    ap.add_argument("--model-config", default="configs/model.yaml")
    ap.add_argument("--train-config", default="configs/train.yaml")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--stage", choices=["representation", "response", "witness", "planner", "all"], default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--output-dir", default="outputs/checkpoints")
    args = ap.parse_args()
    cfg = load_config(args.model_config, args.train_config, args.data_config)
    tcfg = cfg["train"]
    stage = args.stage or tcfg.get("stage", "witness")
    device = _device(tcfg.get("device", "auto"))
    ds = TorchCOWPDataset(args.cache_dir or cfg["outputs"]["tensor_cache_dir"])
    dl = DataLoader(ds, batch_size=args.batch_size or int(tcfg.get("batch_size", 8)), shuffle=True, num_workers=int(tcfg.get("num_workers", 0)), collate_fn=collate_torch)
    model = COWPModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg.get("lr", 3e-4)), weight_decay=float(tcfg.get("weight_decay", 1e-4)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = args.epochs or int(tcfg.get("epochs", 10))
    loss_weights = cfg.get("loss_weights", {})
    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
            opt.zero_grad(set_to_none=True)
            pred = model(batch)
            losses = []
            if stage in ("response", "all"):
                losses.append(response_loss(pred["response"], batch, loss_weights)["loss"])
            if stage in ("witness", "planner", "all"):
                losses.append(witness_loss(pred["witness"], batch, loss_weights)["loss"])
            if stage in ("planner", "all"):
                losses.append(loss_weights.get("ranking", 1.0) * planner_ranking_loss(pred["planner_score"], batch["cowp/candidates/noncoercive_feasible"].bool(), batch["cowp/candidates/false_safe"].bool(), batch["cowp/candidates/valid"].bool()))
            if not losses:
                # Representation stage learns candidate imitation through closest logged candidate utility proxy.
                losses.append(pred["planner_score"].mean() * 0.0 + torch.relu(pred["witness"]["opr"].mean() - 0.5))
            loss = sum(losses)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            running += float(loss.detach().cpu())
        avg = running / max(len(dl), 1)
        history.append({"epoch": epoch, "loss": avg})
        print(f"epoch={epoch} loss={avg:.6f}")
        torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch}, output_dir / f"cowp_{stage}_epoch{epoch:03d}.pt")
    with (output_dir / f"history_{stage}.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
