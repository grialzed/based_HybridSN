"""绘制 Indian Pines 上 HybridSN 的测试混淆矩阵、验证召回率和预测图。

在项目目录运行。metrics 读取已有 CSV，predict 读取权重并推理，
train 复用原训练流程，保存最佳权重和 PCA 参数，并生成这次运行的图。
适用配置为 PCA 30 维、邻域块 25×25。
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType

import matplotlib
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
from matplotlib import pyplot as plt


COMPONENTS = 30
WINDOW = 25
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8))
# 与 display_dataset.py 的印度松树真值图使用相同的 0—16 类色表。
CLASS_COLORS = (
    "#ffffff", "#e6194b", "#3cb44b", "#ffe119", "#4363d8",
    "#f58231", "#911eb4", "#46f0f0", "#f032e6", "#bcf60c",
    "#fabebe", "#008080", "#e6beff", "#9a6324", "#fffac8",
    "#800000", "#aaffc3",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    metrics = commands.add_parser("metrics", help="从已有实验 CSV 绘制测试混淆矩阵和验证召回率")
    metrics.add_argument("--run-dir", type=Path, default=Path("results/hybridsn"), help="包含测试矩阵的结果目录")
    metrics.add_argument("--seed", type=int, default=11, help="选择 test_confusion_seedXX.csv")
    metrics.add_argument("--validation-csv", type=Path, help="验证类别统计，默认使用结果目录下的 pca_classes.csv")

    train = commands.add_parser("train", help="训练一次，保存最佳权重和 PCA，并生成全部图")
    train.add_argument("--seed", type=int, default=11, help="训练种子")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--lr", type=float, default=1e-3)

    predict = commands.add_parser("predict", help="加载权重并把预测图与真值并排绘制")
    predict.add_argument("--weights", type=Path, required=True, help="train 命令的检查点或 PyTorch state_dict")

    for command in (metrics, train, predict):
        command.add_argument("--model-script", type=Path, default=Path("hybridsn_pca_window.py"))
        command.add_argument("--output-dir", type=Path, help="新输出目录，默认在脚本旁生成带时间戳的目录")
    for command in (train, predict):
        command.add_argument("--data-dir", type=Path, default=Path("data"), help="两份 Indian Pines .mat 文件所在目录")
        command.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
        command.add_argument("--batch-size", type=int, default=32)
        command.add_argument("--amp", action="store_true", help="在 CUDA 使用混合精度")
        command.add_argument("--train-ratio", type=float, default=0.10, help="训练集比例，裸权重推理时须与原训练相同")
        command.add_argument("--val-ratio", type=float, default=0.10)
        command.add_argument("--split-seed", type=int, default=345, help="数据划分种子，裸权重推理时须与原训练相同")
        command.add_argument("--full-scene", action="store_true", help="预测全图，默认只显示真值有标签的区域")

    args = parser.parse_args()
    if args.command in ("metrics", "train") and args.seed < 0:
        parser.error("--seed 必须为非负整数。")
    if args.command != "metrics":
        if args.batch_size < 1:
            parser.error("--batch-size 必须为正整数。")
        if not (0 < args.train_ratio and 0 < args.val_ratio and args.train_ratio + args.val_ratio < 1):
            parser.error("训练和验证比例必须为正，且总和小于 1。")
        if not 0 <= args.split_seed < 2**32 - 1:
            parser.error("--split-seed 须满足后续验证和测试的二次划分范围。")
    if args.command == "train" and (args.epochs < 1 or not np.isfinite(args.lr) or args.lr <= 0):
        parser.error("--epochs 和 --lr 必须为正。")
    return args


def load_baseline(path: Path) -> ModuleType:
    if not path.is_file():
        raise FileNotFoundError(f"找不到 HybridSN 源码 {path}，请在项目目录运行或指定 --model-script。")
    spec = importlib.util.spec_from_file_location("_draw_confusion_baseline", path.resolve())
    if spec is None or spec.loader is None:
        raise ValueError(f"无法加载 HybridSN 源码 {path}。")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def new_output_directory(args: argparse.Namespace) -> Path:
    stamp = datetime.now(SHANGHAI_TIMEZONE).strftime("%Y%m%d_%H%M%S_%f")
    default = Path(__file__).resolve().parent / f"draw_confusion_{args.command}_{stamp}"
    directory = (args.output_dir or default).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def read_metrics(
    run_dir: Path, seed: int, validation_csv: Path, names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix_path = run_dir / f"test_confusion_seed{seed}.csv"
    confusion = np.loadtxt(matrix_path, delimiter=",", dtype=np.int64)
    if confusion.shape != (len(names), len(names)) or np.any(confusion < 0) or np.any(confusion.sum(axis=1) == 0):
        raise ValueError(f"{matrix_path} 应为 16×16 非负计数矩阵，每个真实类别须有测试像素。")

    with validation_csv.open(newline="", encoding="utf-8-sig") as file:
        rows = [
            row for row in csv.DictReader(file)
            if int(row["pca_components"]) == COMPONENTS and int(row["window_size"]) == WINDOW
        ]
    rows.sort(key=lambda row: int(row["class_id"]))
    if [int(row["class_id"]) for row in rows] != list(range(1, len(names) + 1)) or tuple(row["class"] for row in rows) != names:
        raise ValueError(f"{validation_csv} 须包含 K={COMPONENTS}、{WINDOW}×{WINDOW} 配置的完整 16 类验证记录。")
    means = np.array([float(row["val_recall_mean_pct"]) for row in rows])
    stds = np.array([float(row["val_recall_sd_pp"]) for row in rows])
    if not np.all((0 <= means) & (means <= 100)) or not np.all(np.isfinite(stds) & (stds >= 0)):
        raise ValueError(f"{validation_csv} 包含无效的召回率或标准差。")
    return confusion, means, stds


def draw_metrics(
    base: ModuleType, confusion: np.ndarray, means: np.ndarray, stds: np.ndarray,
    seed: int, directory: Path,
) -> None:
    names = base.CLASS_NAMES
    positions = np.arange(len(names))
    class_ids = [str(index + 1) for index in positions]
    labels = [f"{index + 1:02d} {name}" for index, name in enumerate(names)]
    row_pct = 100 * confusion / confusion.sum(axis=1, keepdims=True)
    oa, aa = base.scores(confusion)[:2]

    fig, ax = plt.subplots(figsize=(11.2, 8.2), layout="constrained")
    image = ax.imshow(row_pct, cmap="Blues", vmin=0, vmax=100, interpolation="nearest")
    for row, column in zip(*np.nonzero(confusion)):
        color = "white" if row_pct[row, column] >= 58 else "#1c2f40"
        ax.text(column, row, str(confusion[row, column]), ha="center", va="center", color=color, fontsize=9)
    ax.set_xticks(positions, class_ids)
    ax.set_yticks(positions, labels)
    ax.set(
        title=f"Test confusion matrix | seed {seed} | OA {oa:.2f}% | AA {aa:.2f}%",
        xlabel="Predicted class (ID)", ylabel="True class",
    )
    fig.colorbar(image, ax=ax, label="Share of the true class (%)", shrink=0.85)
    base.save_figure(fig, directory / f"test_confusion_seed{seed}.png")

    fig, ax = plt.subplots(figsize=(13.4, 5.6), layout="constrained")
    errors = stds if np.any(stds) else None
    ax.bar(positions, means, yerr=errors, capsize=3, color="#477B9C", zorder=3)
    for index, (mean, sd) in enumerate(zip(means, stds)):
        ax.text(index, mean + sd + 1.0, f"{mean:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(positions, labels, rotation=38, ha="right")
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylim(0, max(106, float(np.max(means + stds)) + 6))
    title = "Validation recall by class" if errors is None else "Validation recall by class (mean ± SD across seeds)"
    ax.set(title=f"{title} | K={COMPONENTS}, {WINDOW}×{WINDOW}", ylabel="Recall (%)")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    base.save_figure(fig, directory / "validation_recall.png")


def select_device(args: argparse.Namespace) -> torch.device:
    cuda = torch.cuda.is_available()
    name = ("cuda" if cuda else "cpu") if args.device == "auto" else args.device
    device = torch.device(name)
    if device.type == "cuda" and not cuda:
        raise ValueError("当前 PyTorch 环境未检测到 CUDA。")
    if args.amp and device.type != "cuda":
        raise ValueError("--amp 需要 CUDA。")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    return device


def fit_scene_pca(
    base: ModuleType, data: np.ndarray, labels: np.ndarray, args: argparse.Namespace,
):
    split = base.split_pixels(labels, args)
    spectra = data.reshape(-1, data.shape[-1])
    pca = base.PCA(svd_solver="full", whiten=True).fit(spectra[split["train"]])
    if pca.n_components_ < COMPONENTS:
        raise ValueError(f"当前划分的训练光谱须支持至少 {COMPONENTS} 个主成分。")
    return pca, split


def project_scene(pca, data: np.ndarray) -> np.ndarray:
    spectra = data.reshape(-1, data.shape[-1])
    projected = pca.transform(spectra)[:, :COMPONENTS]
    return projected.reshape(*data.shape[:2], COMPONENTS).astype(np.float32)


def restore_pca(base: ModuleType, packed: dict, bands: int):
    mean = packed["mean"].numpy()
    components = packed["components"].numpy()
    variance = packed["explained_variance"].numpy()
    if (
        mean.shape != (bands,) or components.ndim != 2 or components.shape[1] != bands
        or components.shape[0] < COMPONENTS or variance.shape != (components.shape[0],)
    ):
        raise ValueError("检查点中的 PCA 参数尺寸与输入光谱不匹配。")
    pca = base.PCA(svd_solver="full", whiten=True)
    pca.mean_ = mean
    pca.components_ = components
    pca.explained_variance_ = variance
    pca.n_features_in_ = bands
    pca.n_components_ = len(components)
    return pca


def predict_pixels(
    base: ModuleType, model, projected: np.ndarray, labels: np.ndarray,
    args: argparse.Namespace, device: torch.device, amp: bool,
) -> np.ndarray:
    setting = base.Setting(COMPONENTS, WINDOW)
    indices = np.arange(labels.size) if args.full_scene else np.flatnonzero(labels.ravel())
    dataset = base.PixelPatches(base.make_padded(projected, setting), labels, indices, WINDOW)
    loader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=device.type == "cuda")
    predictions = []
    with torch.inference_mode():
        for patches, _ in loader:
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(patches.to(device, non_blocking=True))
            predictions.append(logits.argmax(dim=1).cpu().numpy() + 1)
    pixel_map = np.zeros(labels.size, dtype=np.int64)
    pixel_map[indices] = np.concatenate(predictions)
    return pixel_map.reshape(labels.shape)


def render_checkpoint(
    base: ModuleType, weight_path: Path, data: np.ndarray, labels: np.ndarray,
    args: argparse.Namespace, device: torch.device, directory: Path,
) -> None:
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("权重文件应包含 PyTorch state_dict 或 train 命令生成的检查点。")
    if checkpoint.get("pca_components", COMPONENTS) != COMPONENTS or checkpoint.get("window_size", WINDOW) != WINDOW:
        raise ValueError(f"权重需要对应 K={COMPONENTS}、{WINDOW}×{WINDOW} 的 HybridSN。")

    model = base.HybridSN().to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict)
    model.eval()

    if "pca" in checkpoint:
        pca = restore_pca(base, checkpoint["pca"], data.shape[-1])
    else:
        print(f"权重仅含模型参数，按 split-seed={args.split_seed} 和 train-ratio={args.train_ratio:g} 重新拟合 PCA。")
        pca, _ = fit_scene_pca(base, data, labels, args)
    amp = device.type == "cuda" and (args.amp or bool(checkpoint.get("amp", False)))
    predicted = predict_pixels(base, model, project_scene(pca, data), labels, args, device, amp)

    cmap = ListedColormap(CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(CLASS_COLORS) + 0.5), len(CLASS_COLORS))
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 6.8), layout="constrained")
    scope = "full scene" if args.full_scene else "labeled pixels"
    for ax, pixels, title in zip(axes, (labels, predicted), ("Ground truth", f"HybridSN prediction ({scope})")):
        ax.imshow(pixels, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(title)
        ax.set_axis_off()
    seed = checkpoint.get("seed")
    config = f"Indian Pines | K={COMPONENTS}, {WINDOW}×{WINDOW}"
    fig.suptitle(config if seed is None else f"{config} | seed {seed}")
    names = ("Background", *base.CLASS_NAMES)
    legend = [
        Patch(facecolor=color, edgecolor="#aaaaaa", linewidth=0.5, label=f"{idx:02d} {name}")
        for idx, (color, name) in enumerate(zip(CLASS_COLORS, names))
    ]
    fig.legend(handles=legend, loc="outside lower center", ncols=4, fontsize=9, frameon=False)
    base.save_figure(fig, directory / "hybridsn_prediction.png")


def train_and_draw(
    base: ModuleType, data: np.ndarray, labels: np.ndarray,
    args: argparse.Namespace, device: torch.device, directory: Path,
) -> None:
    pca, split = fit_scene_pca(base, data, labels, args)
    projected = project_scene(pca, data)
    setting = base.Setting(COMPONENTS, WINDOW)
    training_args = argparse.Namespace(**vars(args), seeds=[args.seed], final_test=True)
    result = base.run_setting(setting, projected, labels, split, training_args, device)
    trial = result.trials[0]
    weight_path = directory / f"hybridsn_seed{args.seed}.pt"
    if trial.weights is None:
        raise RuntimeError("训练流程未返回验证集最佳轮次的权重。")
    torch.save({
        "model_state_dict": trial.weights,
        "pca": {
            "mean": torch.from_numpy(pca.mean_.copy()),
            "components": torch.from_numpy(pca.components_.copy()),
            "explained_variance": torch.from_numpy(pca.explained_variance_.copy()),
        },
        "pca_components": COMPONENTS,
        "window_size": WINDOW,
        "seed": trial.seed,
        "best_epoch": trial.best_epoch,
        "split_seed": args.split_seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "amp": args.amp,
    }, weight_path)
    print(f"Saved checkpoint {weight_path}", flush=True)

    support = np.bincount(labels.ravel()[split["validation"]], minlength=base.N_CLASSES + 1)[1:]
    cumulative = np.cumsum(100 * pca.explained_variance_ratio_)
    base.export_sweep("pca", [result], result, result, support, cumulative, directory)
    base.final_test(result, projected, labels, split["test"], training_args, device, directory)
    matrix, means, stds = read_metrics(directory, args.seed, directory / "pca_classes.csv", base.CLASS_NAMES)
    draw_metrics(base, matrix, means, stds, args.seed, directory)
    render_checkpoint(base, weight_path, data, labels, args, device, directory)


def main() -> None:
    args = parse_args()
    base = load_baseline(args.model_script)

    if args.command == "metrics":
        validation = args.validation_csv or args.run_dir / "pca_classes.csv"
        matrix, means, stds = read_metrics(args.run_dir, args.seed, validation, base.CLASS_NAMES)
        directory = new_output_directory(args)
        draw_metrics(base, matrix, means, stds, args.seed, directory)
    else:
        if args.command == "predict" and not args.weights.is_file():
            raise FileNotFoundError(f"找不到权重文件 {args.weights}。")
        device = select_device(args)
        data, labels = base.read_scene(args.data_dir)
        directory = new_output_directory(args)
        if args.command == "train":
            train_and_draw(base, data, labels, args, device, directory)
        else:
            render_checkpoint(base, args.weights, data, labels, args, device, directory)
    print(f"Outputs {directory}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"错误 {error}") from None
