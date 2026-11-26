"""Indian Pines 上的 HybridSN PCA 维数与邻域大小实验。

依赖：Python 3.9+，NumPy，SciPy，scikit-learn，PyTorch 2.3+，Matplotlib。
数据：同一 --data-dir 下的 Indian_pines_corrected.mat 和 Indian_pines_gt.mat。

先看 PCA 能量：python hybridsn_pca_window.py --mode variance
固定窗口比较 PCA：python hybridsn_pca_window.py --mode pca --fixed-window 25
固定 PCA 比较窗口：python hybridsn_pca_window.py --mode window --fixed-components 30
依次进行两组对照：python hybridsn_pca_window.py --mode all
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import sklearn
import torch
from scipy.io import loadmat
from scipy.ndimage import maximum_filter, uniform_filter
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
from matplotlib import pyplot as plt

CLASS_NAMES = (
    "Alfalfa", "Corn-notill", "Corn-mintill", "Corn",
    "Grass-pasture", "Grass-trees", "Grass-pasture-mowed", "Hay-windrowed",
    "Oats", "Soybean-notill", "Soybean-mintill", "Soybean-clean",
    "Wheat", "Woods", "Buildings-Grass-Trees-Drives", "Stone-Steel-Towers",
)
N_CLASSES = len(CLASS_NAMES)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR.parent / "downloads" if SCRIPT_DIR.name == "output" else Path.cwd()


@dataclass(frozen=True)
class Setting:
    components: int
    window: int


@dataclass
class Trial:
    seed: int
    best_epoch: int
    confusion: np.ndarray
    train_seconds_per_epoch: float
    weights: dict[str, torch.Tensor] | None


@dataclass
class Result:
    setting: Setting
    parameters: int
    overlap_pct: float
    trials: list[Trial]
    means: np.ndarray
    stds: np.ndarray

class HybridSN(nn.Module):
  def __init__(self, in_channels=1, out_channels=16, window_size=25, pca_components=30):
    super(HybridSN, self).__init__()
    self.conv3d_features = nn.Sequential(
        nn.Conv3d(in_channels,out_channels=8,kernel_size=(7,3,3)),
        nn.ReLU(),
        nn.Conv3d(in_channels=8,out_channels=16,kernel_size=(5,3,3)),
        nn.ReLU(),
        nn.Conv3d(in_channels=16,out_channels=32,kernel_size=(3,3,3)),
        nn.ReLU()
    )

    self.conv2d_features = nn.Sequential(
        nn.Conv2d(in_channels=32 * (pca_components - 12), out_channels=64, kernel_size=(3,3)),
        nn.ReLU()
    )

    self.classifier = nn.Sequential(
        nn.Linear(64 * (window_size - 8) ** 2, 256),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Dropout(p=0.4),
        nn.Linear(128, 16)
    )

  def forward(self, x):
    x = self.conv3d_features(x)
    x = x.view(x.size()[0],x.size()[1]*x.size()[2],x.size()[3],x.size()[4])
    x = self.conv2d_features(x)
    x = x.view(x.size()[0],-1)
    x = self.classifier(x)
    return x


class PixelPatches(Dataset):
    """按中心坐标即时取块，使各候选共享样本且无需一次性保存所有像素块。"""

    def __init__(self, padded: torch.Tensor, labels: np.ndarray, pixels: np.ndarray, width: int):
        self.padded = padded
        self.rows, self.cols = np.unravel_index(pixels, labels.shape)
        self.targets = torch.as_tensor(labels.ravel()[pixels] - 1, dtype=torch.long)
        self.width = width

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row, col, width = self.rows[index], self.cols[index], self.width
        patch = self.padded[:, row:row + width, col:col + width]
        return patch.unsqueeze(0), self.targets[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("variance", "pca", "window", "all"), default="all",
                        help="只看能量、只扫 PCA、只扫窗口，或先扫 PCA 再扫窗口")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, help="新结果目录。默认在脚本旁生成带时间戳的目录")
    parser.add_argument("--pca-components", type=int, nargs="+", default=[15, 30, 40])
    parser.add_argument("--window-sizes", type=int, nargs="+", default=[15, 19, 25])
    parser.add_argument("--fixed-window", type=int, default=25, help="PCA 对照的统一窗口边长")
    parser.add_argument("--fixed-components", type=int, default=30, help="window 模式的统一 PCA 主成分数")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--split-seed", type=int, default=345)
    parser.add_argument("--train-ratio", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--aa-tolerance", type=float, default=0.5,
                        help="距最高验证 AA 多少个百分点以内可优先选择参数较少的一组")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--amp", action="store_true", help="在 CUDA 上开启混合精度")
    parser.add_argument("--final-test", action="store_true",
                        help="仅对本次选择的配置评估留出的测试集")
    args = parser.parse_args()
    args.pca_components = sorted(set(args.pca_components))
    args.window_sizes = sorted(set(args.window_sizes))
    if min(args.epochs, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("epochs、batch-size 和 lr 必须为正数")
    if not (args.train_ratio > 0 and args.val_ratio > 0 and args.train_ratio + args.val_ratio < 1):
        parser.error("train-ratio 和 val-ratio 必须为正，且两者之和小于 1")
    if len(set(args.seeds)) != len(args.seeds) or min([args.split_seed, *args.seeds]) < 0:
        parser.error("训练种子应互不重复，所有种子须为非负整数")
    if not 0 <= args.aa_tolerance <= 100:
        parser.error("aa-tolerance 的单位是百分点，范围为 0 到 100")
    if args.mode == "variance" and args.final_test:
        parser.error("variance 模式只计算 PCA，无法评估测试集")
    return args


def read_scene(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(folder / "Indian_pines_corrected.mat")["indian_pines_corrected"]
    labels = loadmat(folder / "Indian_pines_gt.mat")["indian_pines_gt"].astype(np.int64)
    if data.shape != (145, 145, 200) or labels.shape != data.shape[:2]:
        raise ValueError("请输入 Indian Pines corrected 场景及对应的 145 × 145 真值图。")
    if not np.array_equal(np.unique(labels), np.arange(N_CLASSES + 1)):
        raise ValueError("真值图应包含背景 0 和 16 个类别标签 1—16。")
    return data, labels


def split_pixels(labels: np.ndarray, args: argparse.Namespace) -> dict[str, np.ndarray]:
    flat_labels = labels.ravel()
    labeled = np.flatnonzero(flat_labels)
    train, rest = train_test_split(
        labeled, train_size=args.train_ratio, stratify=flat_labels[labeled], random_state=args.split_seed,
    )
    val, test = train_test_split(
        rest, train_size=args.val_ratio / (1 - args.train_ratio),
        stratify=flat_labels[rest], random_state=args.split_seed + 1,
    )
    split = {"train": train, "validation": val, "test": test}
    for name, indices in split.items():
        if len(np.unique(flat_labels[indices])) != N_CLASSES:
            raise ValueError(f"{name} 缺少类别，请调整划分比例或 split-seed。")
    return split


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def export_energy(pca: PCA, candidates: list[int], directory: Path) -> np.ndarray:
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    energy = 100 * pca.explained_variance_ratio_
    cumulative = np.cumsum(energy)
    write_csv(directory / "pca_energy.csv", [
        {"component": index, "explained_variance_pct": ratio, "cumulative_variance_pct": total}
        for index, ratio, total in zip(components, energy, cumulative)
    ])

    fig, (individual, combined) = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    last = max(30, max(candidates))
    idx = components <= last
    individual.bar(components[idx], energy[idx], color="#477B9C")
    individual.set(title="Individual explained variance", xlabel="Principal component", ylabel="Variance (%)")
    combined.plot(components, cumulative, color="#185D68", linewidth=2)
    for value in candidates:
        combined.scatter(value, cumulative[value - 1], s=24, zorder=3,
                         label=f"K={value}: {cumulative[value - 1]:.2f}%")
    combined.set(title="Cumulative explained variance", xlabel="Number of components", ylabel="Variance (%)")
    combined.legend(loc="lower right", frameon=False)
    for ax in (individual, combined):
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    fig.suptitle("Indian Pines | PCA fitted on training spectra")
    save_figure(fig, directory / "pca_energy.png")
    return cumulative


def patch_overlap(shape: tuple[int, int], train: np.ndarray, evaluated: np.ndarray, window: int) -> float:
    """评估块中与任意训练块共享的真实图像像素比例，对评估样本取平均。"""
    centers = np.zeros(shape, dtype=np.float64)
    centers.flat[train] = 1
    seen_by_train = maximum_filter(centers, size=window, mode="constant")
    shared = uniform_filter(seen_by_train, size=window, mode="constant")
    real = uniform_filter(np.ones(shape), size=window, mode="constant")
    ratios = shared.ravel()[evaluated] / real.ravel()[evaluated]
    return float(100 * np.clip(ratios, 0, 1).mean())


def make_padded(projected: np.ndarray, setting: Setting) -> torch.Tensor:
    radius = setting.window // 2
    channels_first = projected[..., :setting.components].transpose(2, 0, 1)
    padded = np.pad(channels_first, ((0, 0), (radius, radius), (radius, radius)))
    return torch.from_numpy(np.ascontiguousarray(padded))


def scores(confusion: np.ndarray) -> np.ndarray:
    correct = np.diag(confusion)
    recalls = 100 * correct / confusion.sum(axis=1)
    return np.r_[100 * correct.sum() / confusion.sum(), recalls.mean(), recalls]


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, amp: bool) -> np.ndarray:
    model.eval()
    counts = np.zeros(N_CLASSES * N_CLASSES, dtype=np.int64)
    for x, labels in loader:
        with torch.autocast(device_type=device.type, enabled=amp):
            pred = model(x.to(device, non_blocking=True)).argmax(dim=1).cpu()
        counts += np.bincount((labels * N_CLASSES + pred).numpy(), minlength=counts.size)
    return counts.reshape(N_CLASSES, N_CLASSES)


def fit(
    setting: Setting, training: PixelPatches, validation: DataLoader,
    args: argparse.Namespace, device: torch.device, seed: int,
) -> tuple[Trial, int]:
    torch.manual_seed(seed)
    model = HybridSN(window_size=setting.window, pca_components=setting.components).to(device)
    parameters = sum(param.numel() for param in model.parameters() if param.requires_grad)
    train_loader = DataLoader(
        training, batch_size=args.batch_size, shuffle=True, pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    best_key = (-1.0, -1.0)
    best_weights = None
    train_seconds = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        start = time.perf_counter()
        for x, labels in train_loader:
            x = x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=args.amp):
                loss = loss_fn(model(x), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds += time.perf_counter() - start

        confusion = evaluate(model, validation, device, args.amp)
        oa, aa = scores(confusion)[:2]
        if (aa, oa) > best_key:
            best_key, best_epoch, best_confusion = (aa, oa), epoch, confusion
            if args.final_test:
                best_weights = {key: tensor.detach().to("cpu", copy=True)
                                for key, tensor in model.state_dict().items()}
        if epoch in (1, args.epochs) or epoch % 10 == 0:
            print(f"  seed={seed}, epoch={epoch:3d}/{args.epochs}, val OA={oa:.2f}%, AA={aa:.2f}%", flush=True)
    return Trial(seed, best_epoch, best_confusion, train_seconds / args.epochs, best_weights), parameters


def run_setting(
    setting: Setting, projected: np.ndarray, labels: np.ndarray, split: dict[str, np.ndarray],
    args: argparse.Namespace, device: torch.device,
) -> Result:
    print(f"Training K={setting.components}, window={setting.window}x{setting.window}", flush=True)
    padded = make_padded(projected, setting)
    training = PixelPatches(padded, labels, split["train"], setting.window)
    validation = DataLoader(
        PixelPatches(padded, labels, split["validation"], setting.window),
        batch_size=args.batch_size, pin_memory=device.type == "cuda",
    )
    trials = []
    for seed in args.seeds:
        trial, parameters = fit(setting, training, validation, args, device, seed)
        trials.append(trial)
    metrics = np.stack([scores(trial.confusion) for trial in trials])
    stds = metrics.std(axis=0, ddof=1) if len(trials) > 1 else np.zeros(metrics.shape[1])
    overlap = patch_overlap(labels.shape, split["train"], split["validation"], setting.window)
    return Result(setting, parameters, overlap, trials, metrics.mean(axis=0), stds)


def choose(results: list[Result], tolerance: float) -> tuple[Result, Result]:
    # 指标向量顺序是 OA、AA、16 类召回率。先确定最高 AA，再在容差内比较参数量。
    best = max(results, key=lambda row: (row.means[1], row.means[0], -row.parameters))
    near_best = (row for row in results if row.means[1] >= best.means[1] - tolerance)
    selected = min(near_best, key=lambda row: (row.parameters, -row.means[1], -row.means[0]))
    return best, selected


def train_seconds(result: Result) -> float:
    return float(np.mean([trial.train_seconds_per_epoch for trial in result.trials]))


def export_sweep(
    mode: str, results: list[Result], best: Result, selected: Result,
    validation_support: np.ndarray, cumulative: np.ndarray, directory: Path,
) -> None:
    summary = []
    class_rows = []
    trial_rows = []
    for row in results:
        config = {"pca_components": row.setting.components, "window_size": row.setting.window}
        summary.append({
            **config,
            "cumulative_variance_pct": cumulative[row.setting.components - 1],
            "val_oa_mean_pct": row.means[0], "val_oa_sd_pp": row.stds[0],
            "val_aa_mean_pct": row.means[1], "val_aa_sd_pp": row.stds[1],
            "parameters": row.parameters, "train_seconds_per_epoch": train_seconds(row),
            "val_patch_overlap_pct": row.overlap_pct, "seed_count": len(row.trials),
            "highest_aa": row is best, "recommended": row is selected,
        })
        for i, name in enumerate(CLASS_NAMES):
            class_rows.append({
                **config, "class_id": i + 1, "class": name, "validation_pixels": validation_support[i],
                "val_recall_mean_pct": row.means[i + 2], "val_recall_sd_pp": row.stds[i + 2],
            })
        for trial in row.trials:
            oa, aa = scores(trial.confusion)[:2]
            trial_rows.append({
                **config, "seed": trial.seed, "best_epoch": trial.best_epoch,
                "val_oa_pct": oa, "val_aa_pct": aa, "train_seconds_per_epoch": trial.train_seconds_per_epoch,
            })
    write_csv(directory / f"{mode}_summary.csv", summary)
    write_csv(directory / f"{mode}_classes.csv", class_rows)
    write_csv(directory / f"{mode}_trials.csv", trial_rows)


def draw_sweep(
    mode: str, results: list[Result], selected: Result,
    support: np.ndarray, directory: Path,
) -> None:
    means = np.stack([row.means for row in results])
    stds = np.stack([row.stds for row in results])
    is_pca = mode == "pca"
    x = [row.setting.components if is_pca else row.setting.window for row in results]
    x_label = "PCA components" if is_pca else "Patch side length (pixels)"
    chosen = selected.setting.components if is_pca else selected.setting.window
    trial_count = len(selected.trials)
    error_bars = trial_count > 1
    statistics = f"mean ± SD ({trial_count} seeds)" if error_bars else "1 seed"

    if is_pca:
        fig, accuracy = plt.subplots(figsize=(7.3, 4.2), layout="constrained")
        figure_name = "pca_overall_accuracy.png"
    else:
        fig, axes = plt.subplots(2, 2, figsize=(10.3, 6.5), layout="constrained")
        accuracy, *other_axes = list(axes.flat)
        figure_name = "window_tradeoff.png"
        plots = (
            ([row.parameters / 1e6 for row in results], "Trainable parameters", "Millions"),
            ([train_seconds(row) for row in results], "Training time", "Seconds per epoch"),
            ([row.overlap_pct for row in results], "Validation patch overlap", "Shared real pixels (%)"),
        )
        for ax, (values, title, unit) in zip(other_axes, plots):
            ax.plot(x, values, "-o", color="#477B9C")
            ax.set(title=title, xlabel=x_label, ylabel=unit, xticks=x)
            ax.grid(axis="y", alpha=0.25)
            ax.axvline(chosen, color="0.68", linestyle=":", zorder=1)

    for column, color, label in ((0, "#185D68", "Overall accuracy (OA)"), (1, "#B4532F", "Average accuracy (AA)")):
        accuracy.errorbar(x, means[:, column], yerr=stds[:, column] if error_bars else None,
                          fmt="-o", capsize=3, color=color, label=label)
    accuracy.axvline(chosen, color="0.68", linestyle=":", label=f"Selected: {chosen}")
    accuracy.set(title="Validation accuracy", xlabel=x_label, ylabel="Accuracy (%)", xticks=x)
    accuracy.legend(frameon=False)
    accuracy.grid(axis="y", alpha=0.25)
    condition = f"fixed {selected.setting.window} x {selected.setting.window} patch" if is_pca else f"fixed K={selected.setting.components}"
    fig.suptitle(f"Indian Pines | {condition} | {statistics}")
    save_figure(fig, directory / figure_name)

    class_means = means[:, 2:]
    class_stds = stds[:, 2:]
    recall_min = min(-4, float((class_means - class_stds).min()) - 3)
    recall_max = max(104, float((class_means + class_stds).max()) + 3)
    fig, axes = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12, 7.8), layout="constrained")
    for i, ax in enumerate(axes.flat):
        ax.errorbar(x, means[:, i + 2], yerr=stds[:, i + 2] if error_bars else None,
                    fmt="-o", color="#185D68", markersize=4, capsize=2)
        ax.axvline(chosen, color="0.68", linestyle=":", zorder=1)
        name = CLASS_NAMES[i].replace("Buildings-Grass-Trees-Drives", "Buildings / vegetation / drives")
        ax.set(title=f"{i + 1:02d} {name}\nn={support[i]}", xticks=x, yticks=[0, 50, 100], ylim=(recall_min, recall_max))
        ax.title.set_fontsize(9)
        ax.grid(axis="y", alpha=0.25)
    fig.supxlabel(x_label)
    fig.supylabel("Per-class validation accuracy (recall, %)")
    fig.suptitle(f"Indian Pines | {condition} | {statistics}", fontsize=13)
    save_figure(fig, directory / f"{mode}_class_accuracy.png")


def final_test(
    selected: Result, projected: np.ndarray, labels: np.ndarray, indices: np.ndarray,
    args: argparse.Namespace, device: torch.device, directory: Path,
) -> None:
    setting = selected.setting
    test_loader = DataLoader(
        PixelPatches(make_padded(projected, setting), labels, indices, setting.window),
        batch_size=args.batch_size, pin_memory=device.type == "cuda",
    )
    model = HybridSN(window_size=setting.window,pca_components=setting.components).to(device)
    rows = []
    for trial in selected.trials:
        model.load_state_dict(trial.weights)
        confusion = evaluate(model, test_loader, device, args.amp)
        oa, aa, *recall = scores(confusion)
        rows.append({
            "pca_components": setting.components, "window_size": setting.window,
            "seed": trial.seed, "epoch_selected_on_validation": trial.best_epoch,
            "test_oa_pct": oa, "test_aa_pct": aa, **dict(zip(CLASS_NAMES, recall)),
        })
        np.savetxt(directory / f"test_confusion_seed{trial.seed}.csv", confusion, fmt="%d", delimiter=",")
        print(f"Final test seed={trial.seed}: OA={oa:.2f}%, AA={aa:.2f}%", flush=True)
    write_csv(directory / "final_test.csv", rows)


def main() -> None:
    args = parse_args()
    data, labels = read_scene(args.data_dir)
    split = split_pixels(labels, args)
    class_sizes = {
        name: np.bincount(labels.ravel()[indices], minlength=N_CLASSES + 1)[1:]
        for name, indices in split.items()
    }
    spectra = data.reshape(-1, data.shape[-1])
    # 一组训练光谱拟合完整 PCA。不同候选共用这一组主轴，只截取前 K 个分量。
    pca = PCA(svd_solver="full", whiten=True).fit(spectra[split["train"]])
    candidates = [args.fixed_components] if args.mode == "window" else args.pca_components
    min_k = 1 if args.mode == "variance" else 13
    if min(candidates) < min_k or max(candidates) > pca.n_components_:
        raise ValueError(f"当前模式的 PCA 主成分数范围为 {min_k}—{pca.n_components_}。")
    windows = []
    if args.mode in ("pca", "all"):
        windows.append(args.fixed_window)
    if args.mode in ("window", "all"):
        windows.extend(args.window_sizes)
    if any(w < 9 or w % 2 == 0 or w > min(labels.shape) for w in windows):
        raise ValueError("窗口边长须为不小于 9 且不超过图像边长的奇数。")

    directory = args.output_dir or SCRIPT_DIR / datetime.now().strftime("hybridsn_%Y%m%d_%H%M%S_%f")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    print("Outputs:", directory, flush=True)
    print("Split counts:", {name: len(indices) for name, indices in split.items()}, flush=True)
    write_csv(directory / "split_counts.csv", [
        {"class_id": i + 1, "class": name, **{key: sizes[i] for key, sizes in class_sizes.items()}}
        for i, name in enumerate(CLASS_NAMES)
    ])
    cumulative = export_energy(pca, candidates, directory)
    for k in candidates:
        print(f"K={k}: cumulative explained variance {cumulative[k - 1]:.4f}%", flush=True)
    if args.mode == "variance":
        return

    name = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 环境未检测到 CUDA。")
    device = torch.device(name)
    if args.amp and device.type != "cuda":
        raise ValueError("--amp 需要 CUDA。")
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device_details = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("Device:", device_details, flush=True)
    metadata = {
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "device": device_details, "torch": torch.__version__, "scikit_learn": sklearn.__version__,
        "data_shape": data.shape, "split": "stratified centers from one scene",
        "selection": "highest validation AA; then fewest parameters within the AA tolerance in percentage points",
    }
    with (directory / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    max_k = max(candidates)
    projected = pca.transform(spectra)[:, :max_k].reshape(*labels.shape, max_k).astype(np.float32)
    modes = ("pca", "window") if args.mode == "all" else (args.mode,)
    fixed_components = args.fixed_components
    evaluated = {}
    for mode in modes:
        configs = ([Setting(k, args.fixed_window) for k in candidates] if mode == "pca"
                   else [Setting(fixed_components, w) for w in args.window_sizes])
        results = []
        for setting in configs:
            if setting not in evaluated:
                evaluated[setting] = run_setting(setting, projected, labels, split, args, device)
            results.append(evaluated[setting])
        best, selected = choose(results, args.aa_tolerance)
        fixed_components = selected.setting.components
        export_sweep(mode, results, best, selected, class_sizes["validation"], cumulative, directory)
        draw_sweep(mode, results, selected, class_sizes["validation"], directory)
        print(f"{mode}: selected K={fixed_components}, window={selected.setting.window}, "
              f"val AA={selected.means[1]:.2f}% (highest {best.means[1]:.2f}%)", flush=True)

    if args.final_test:
        final_test(selected, projected, labels, split["test"], args, device, directory)
    print("Completed:", directory, flush=True)


if __name__ == "__main__":
    main()
