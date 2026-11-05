"""A small CNN star/galaxy classifier on SE++ cutouts, compared with SPREAD_MODEL, CLASS_STAR and MU_MAX.

  python cnn.py train     trains on the injected sources (cnn_dataset.py injected) of the TRAIN_IMAGES
                          CCDs and writes output/cnn/model.pt
  python cnn.py compare   scores every method on the injected sources of the held-out CCDs (exact labels)
                          and on the real Gaia / SDSS-labeled sources (cnn_dataset.py real), times CNN
                          inference, and writes output/cnn/comparison.{json,png} and real_cnn.npz

The CNN never sees real labels, so the real-source scores are out-of-sample on all CCDs. Methods:
SPREAD_MODEL point source = EXTENDED_CLASS 0-1, CLASS_STAR >= 0.5, MU_MAX mu_class 0, rule = SPREAD_MODEL
with MU_MAX for saturated sources, CNN p(star) >= 0.5.
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

TEST_IMAGES = ("SUPA0155974p_ccd8", "SUPA0155974p_ccd9")
MAG_BINS = [(19, 21), (21, 22), (22, 23), (23, 24), (24, 25)]
REAL_MAG_BINS = [(14, 21), (21, 22), (22, 23), (23, 24.5)]
METHOD_COLORS = {"CNN": "k", "SPREAD_MODEL": "tab:blue", "CLASS_STAR": "tab:orange", "MU_MAX": "tab:green",
                 "rule": "tab:purple"}


def conv(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class StarGalaxyCNN(nn.Module):
    """32x32 cutout (2 channels: asinh S/N, PSF) -> logits (galaxy, star); ~100k parameters, ~15 MFLOP."""

    def __init__(self, channels=2, width=16):
        super().__init__()
        w = width
        self.features = nn.Sequential(conv(channels, w), conv(w, w), nn.MaxPool2d(2),
                                      conv(w, 2 * w), conv(2 * w, 2 * w), nn.MaxPool2d(2),
                                      conv(2 * w, 4 * w), nn.MaxPool2d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(4 * w * 4 * 4, 64), nn.ReLU(inplace=True),
                                  nn.Dropout(0.2), nn.Linear(64, 2))

    def forward(self, x):
        return self.head(self.features(x))


def load_injected(output: Path):
    parts = [np.load(p) for p in sorted((output / "cnn").glob("injected_r*.npz"))]
    keys = [k for k in parts[0].files if k != "timing"]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}, len(parts)


def augment(x):
    """Random 90-degree rotation and flip, the same for the whole batch (cheap; batches are shuffled)."""
    x = torch.rot90(x, int(torch.randint(4, ())), dims=(2, 3))
    return torch.flip(x, dims=(3,)) if torch.rand(()) < 0.5 else x


def predict(model, stamps, batch=2048):
    model.eval()
    out = np.empty(len(stamps), np.float32)
    with torch.inference_mode():
        for lo in range(0, len(stamps), batch):
            x = torch.from_numpy(stamps[lo:lo + batch].astype(np.float32))
            out[lo:lo + batch] = torch.softmax(model(x), 1)[:, 1].numpy()
    return out


def train(args):
    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    data, n_real = load_injected(args.output)
    is_test = np.isin(data["image"], TEST_IMAGES)
    rng = np.random.default_rng(0)
    train_idx = np.flatnonzero(~is_test)
    rng.shuffle(train_idx)
    n_val = len(train_idx) // 10
    val_idx, train_idx = train_idx[:n_val], train_idx[n_val:]
    print(f"{n_real} injection realizations: {len(train_idx)} train, {len(val_idx)} validation "
          f"(CCDs other than {', '.join(TEST_IMAGES)}), {is_test.sum()} held-out test")

    x = torch.from_numpy(data["stamps"])
    y = torch.from_numpy(data["label"].astype(np.int64))
    model = StarGalaxyCNN()
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * (len(train_idx) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps)
    loss_fn = nn.CrossEntropyLoss()
    best, best_state, history = np.inf, None, []
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.from_numpy(rng.permutation(train_idx))
        total = 0.0
        for lo in range(0, len(perm) - args.batch + 1, args.batch):
            b = perm[lo:lo + args.batch]
            loss = loss_fn(model(augment(x[b].float())), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(b)
        p = predict(model, data["stamps"][val_idx])
        yv = data["label"][val_idx]
        val_loss = float(-np.mean(np.log(np.clip(np.where(yv == 1, p, 1 - p), 1e-7, 1))))
        val_acc = float(np.mean((p >= 0.5) == (yv == 1)))
        history.append({"epoch": epoch + 1, "train_loss": total / len(perm), "val_loss": val_loss, "val_acc": val_acc})
        print(f"epoch {epoch + 1:2d}: train loss {total / len(perm):.4f}, validation loss {val_loss:.4f}, "
              f"accuracy {val_acc:.2%} ({time.perf_counter() - start:.0f} s)")
        if val_loss < best:
            best, best_state = val_loss, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    train_seconds = time.perf_counter() - start
    torch.save({"state_dict": model.state_dict(), "history": history, "train_seconds": train_seconds,
                "n_train": len(train_idx), "threads": args.threads}, args.output / "cnn" / "model.pt")
    print(f"trained in {train_seconds:.0f} s on {args.threads} threads; wrote {args.output / 'cnn' / 'model.pt'}")


def auc(score, positive):
    """ROC AUC (probability a random positive scores above a random negative), ties averaged."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    ok = np.isfinite(score)
    score, positive = score[ok], positive[ok]
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_scores = score[order]
    _, first, counts = np.unique(sorted_scores, return_index=True, return_counts=True)
    avg = first + (counts - 1) / 2.0
    ranks[order] = np.repeat(avg, counts) + 1
    n_pos, n_neg = positive.sum(), (~positive).sum()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)) if n_pos and n_neg else np.nan


def method_calls(d, p_star):
    """Per method: (point-source call, defined mask, continuous star score)."""
    ec, cs, mc = d["extended_class"], d["class_star"], d["mu_class"]
    sat = d["saturated"]
    spread_point = (ec >= 0) & (ec <= 1)
    mu_point = mc == 0
    return {
        "CNN": (p_star >= 0.5, np.isfinite(p_star), p_star),
        "SPREAD_MODEL": (spread_point, ec >= 0, -d["spread_model"]),
        "CLASS_STAR": (cs >= 0.5, np.isfinite(cs), cs),
        "MU_MAX": (mu_point, mc >= 0, -d["mu_locus_offset"]),
        "rule": (np.where(sat, mu_point, spread_point), np.where(sat, mc >= 0, ec >= 0),
                 np.where(sat, -d["mu_locus_offset"], -d["spread_model"])),
    }


def score_table(label, mag, calls, bins, select):
    """{method: [{bin, n, accuracy, star_ok, galaxy_ok, auc}]} over sources with label 0/1 and select."""
    out = {}
    for name, (point, defined, score) in calls.items():
        rows = []
        for lo, hi in bins:
            b = select & (label >= 0) & (mag >= lo) & (mag < hi)
            star, gal = b & (label == 1), b & (label == 0)
            ok = b & defined
            correct = np.where(label == 1, point, ~point)
            rows.append({"bin": [lo, hi], "n": int(b.sum()), "n_star": int(star.sum()), "n_galaxy": int(gal.sum()),
                         "accuracy": float(np.mean(correct[ok])) if ok.any() else None,
                         "star_ok": float(np.mean(point[star & defined])) if (star & defined).any() else None,
                         "galaxy_ok": float(np.mean(~point[gal & defined])) if (gal & defined).any() else None,
                         "auc": auc(score[b], label[b] == 1) if b.any() else None})
        out[name] = rows
    return out


def print_table(title, table):
    print(f"\n{title}")
    names = list(table)
    print("  mag        n_star n_gal | " + " | ".join(f"{n:>12s} acc   AUC" for n in names))
    for i, row in enumerate(table[names[0]]):
        lo, hi = row["bin"]
        cells = []
        for n in names:
            r = table[n][i]
            acc = f"{r['accuracy']:6.1%}" if r["accuracy"] is not None else "    - "
            a = f"{r['auc']:.3f}" if r["auc"] is not None and np.isfinite(r["auc"]) else "  -  "
            cells.append(f"{'':>12s}{acc} {a}")
        print(f"  {lo:4.1f}-{hi:4.1f}  {row['n_star']:5d} {row['n_galaxy']:5d} | " + " | ".join(cells))


def time_inference(model, stamps, threads_list, repeats=3):
    """Best-of-repeats wall-clock ms per source for CNN inference on all stamps, per thread count."""
    out = {}
    predict(model, stamps[:4096])
    for threads in threads_list:
        torch.set_num_threads(threads)
        best = np.inf
        for _ in range(repeats):
            start = time.perf_counter()
            predict(model, stamps)
            best = min(best, time.perf_counter() - start)
        out[threads] = 1e3 * best / len(stamps)
        print(f"CNN inference, {threads:2d} threads: {out[threads]:.4f} ms per source "
              f"({best:.2f} s for {len(stamps)} cutouts)")
    return out


def plot(results, path):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for ax, key, bins, title in [
            (axes[0], "injected_test", MAG_BINS, "injected sources, held-out CCDs (exact labels)"),
            (axes[1], "real_unsaturated", REAL_MAG_BINS, "real sources, Gaia / SDSS labels (unsaturated)")]:
        for name, rows in results[key].items():
            rows = [r for r in rows if tuple(r["bin"]) in bins]
            mids = [np.mean(r["bin"]) for r in rows]
            acc = [r["accuracy"] if r["accuracy"] is not None else np.nan for r in rows]
            ax.plot(mids, acc, "o-", color=METHOD_COLORS[name], label=name, lw=2 if name == "CNN" else 1.2)
        ax.set(xlabel="auto_mag (SE++)", ylabel="accuracy", ylim=(0.4, 1.01), title=title)
        ax.grid(alpha=0.3)
        ax.legend()

    ax = axes[2]
    timing = results["timing_ms_per_source"]
    labels, values, colors = [], [], []
    for name, color in [("CNN (cutouts + inference)", "k"), ("SPREAD_MODEL", "tab:blue"),
                        ("CLASS_STAR", "tab:orange"), ("MU_MAX", "tab:green")]:
        for jobs, ms in timing[name].items():
            labels.append(f"{name}\n{jobs} CPU")
            values.append(ms)
            colors.append(color)
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(min(values) / 2, max(values) * 4)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.invert_yaxis()
    for i, v in enumerate(values):
        ax.text(v * 1.1, i, f"{v:.3g} ms", va="center", fontsize=8)
    ax.set(xlabel="wall-clock ms per source (log)", title="speed on this laptop (Ryzen 7 260, CPU only)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare(args):
    ckpt = torch.load(args.output / "cnn" / "model.pt", weights_only=False)
    model = StarGalaxyCNN()
    model.load_state_dict(ckpt["state_dict"])
    torch.set_num_threads(args.threads)

    inj, _ = load_injected(args.output)
    test = np.isin(inj["image"], TEST_IMAGES)
    inj = {k: v[test] for k, v in inj.items()}
    p_inj = predict(model, inj["stamps"])
    inj_table = score_table(inj["label"], inj["auto_mag"], method_calls(inj, p_inj), MAG_BINS + [(19, 25)],
                            np.ones(len(p_inj), bool))
    print_table(f"Injected sources on held-out {', '.join(TEST_IMAGES)} ({len(p_inj)}; exact labels)", inj_table)

    real = dict(np.load(args.output / "cnn" / "real.npz"))
    p_real = predict(model, real["stamps"])
    calls = method_calls(real, p_real)
    unsat = score_table(real["label"], real["auto_mag"], calls, REAL_MAG_BINS + [(14, 24.5)], ~real["saturated"])
    print_table("Real sources, Gaia / SDSS labels, unsaturated (CNN never trained on real data)", unsat)
    sat = score_table(real["label"], real["auto_mag"], calls, [(0, 99)], real["saturated"])
    print_table("Real saturated sources (outside the CNN's training range)", sat)
    gaia = real["gaia"] & ~real["saturated"]
    gaia_table = score_table(real["label"], real["auto_mag"], calls, [(18, 20), (20, 22)], gaia)
    print_table("Real Gaia-confirmed stars, unsaturated (fraction called point source)", gaia_table)

    np.savez(args.output / "cnn" / "real_cnn.npz", image=real["image"], source_id=real["source_id"], p_star=p_real)

    print("\nTiming (wall clock):")
    cnn_ms = time_inference(model, real["stamps"], args.threads_list)
    classic = json.loads((args.output / "cnn" / "benchmark_classic.json").read_text())
    stamp_ms = np.median([r["stamps"] for r in classic])
    timing = {"CNN (cutouts + inference)": {t: stamp_ms + ms for t, ms in cnn_ms.items()},
              "SPREAD_MODEL": {r["jobs"]: r["spread_model"] for r in classic},
              "CLASS_STAR": {1: np.median([r["isophotes"] + r["class_star"] for r in classic])},
              "MU_MAX": {1: np.median([r["isophotes"] + r["mu_max"] for r in classic])}}
    print(f"cutouts {stamp_ms:.4f} ms per source (single process)")
    for name, d in timing.items():
        print(f"  {name:26s} " + ", ".join(f"{jobs} CPU: {ms:.4f} ms" for jobs, ms in d.items()))
    print(f"CNN training: {ckpt['train_seconds']:.0f} s on {ckpt['threads']} threads for {ckpt['n_train']} cutouts "
          f"x {len(ckpt['history'])} epochs")

    results = {"injected_test": inj_table, "real_unsaturated": unsat, "real_saturated": sat,
               "real_gaia_unsaturated": gaia_table, "timing_ms_per_source": timing,
               "train_seconds": ckpt["train_seconds"], "history": ckpt["history"]}
    (args.output / "cnn" / "comparison.json").write_text(json.dumps(results, indent=2, default=float))
    plot(results, args.output / "cnn" / "comparison.png")
    print(f"wrote {args.output / 'cnn' / 'comparison.json'} and comparison.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["train", "compare"])
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--threads", type=int, default=8, help="torch threads for training / scoring")
    parser.add_argument("--threads-list", type=int, nargs="+", default=[1, 4, 16], help="thread counts to time")
    args = parser.parse_args()
    train(args) if args.step == "train" else compare(args)


if __name__ == "__main__":
    main()
