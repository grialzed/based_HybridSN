"""根据 Indian Pines 训练光谱的解释方差初选 HybridSN 的 PCA 维数。

读取 --data-dir 中的 Indian_pines_corrected.mat 和 Indian_pines_gt.mat，
输出单项与累计解释方差双图 pca_energy.png 和逐分量数据 pca_energy.csv。
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
from matplotlib import pyplot as plt

SCENE_SHAPE = (145, 145, 200)
N_CLASSES = 16
MIN_HYBRIDSN_COMPONENTS = 13
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default="data", help="两个 .mat 文件所在目录，默认当前工作目录")
    parser.add_argument("--output-dir", type=Path, default="results/pca_select", help="新结果目录，默认在脚本所在目录下以时间戳命名")
    parser.add_argument(
        "--pca-components", type=int, nargs="+", default=[2, 15, 30, 40], metavar="K",
        help="要在图中标注的维数，默认 2 15 30 40，满足 HybridSN 维数下限的项参与初选",
    )
    parser.add_argument("--target-variance", type=float, default=99.0, help="初选所需的累计解释方差百分比，默认 99")
    parser.add_argument("--train-ratio", type=float, default=0.10, help="有标签像素中用于拟合 PCA 的比例，默认 0.10")
    parser.add_argument("--split-seed", type=int, default=345, help="分层划分的随机种子，默认 345")
    args = parser.parse_args()

    args.pca_components = sorted(set(args.pca_components))
    if not 1 <= args.pca_components[0] <= args.pca_components[-1] <= SCENE_SHAPE[-1]:
        parser.error(f"--pca-components 的范围为 1 至 {SCENE_SHAPE[-1]}。")
    if not 0 < args.target_variance <= 100:
        parser.error("--target-variance 应大于 0 且不超过 100。")
    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio 应在 0 和 1 之间。")
    if not 0 <= args.split_seed < 2**32:
        parser.error("--split-seed 应为 0 至 2^32-1 之间的整数。")
    return args


def read_scene(folder: Path) -> tuple[np.ndarray, np.ndarray]:
    data = loadmat(folder / "Indian_pines_corrected.mat")["indian_pines_corrected"]
    labels = loadmat(folder / "Indian_pines_gt.mat")["indian_pines_gt"].astype(np.int64)
    if data.shape != SCENE_SHAPE or labels.shape != data.shape[:2]:
        raise ValueError("请输入 Indian Pines corrected 场景及对应的 145 × 145 真值图。")
    if not np.array_equal(np.unique(labels), np.arange(N_CLASSES + 1)):
        raise ValueError("真值图应包含背景 0 和 16 个类别标签 1—16。")
    return data, labels


def training_spectra(
    data: np.ndarray, labels: np.ndarray, train_ratio: float, split_seed: int,
) -> np.ndarray:
    flat_labels = labels.ravel()
    labeled = np.flatnonzero(flat_labels)
    train_indices, _ = train_test_split(
        labeled, train_size=train_ratio, stratify=flat_labels[labeled], random_state=split_seed,
    )
    if len(np.unique(flat_labels[train_indices])) != N_CLASSES:
        raise ValueError("训练集缺少类别，请调整 --train-ratio 或 --split-seed。")
    return data.reshape(-1, data.shape[-1])[train_indices]


def export_energy(
    individual: np.ndarray, cumulative: np.ndarray, candidates: list[int],
    target: float, selected: int | None, directory: Path,
) -> None:
    components = np.arange(1, len(individual) + 1)
    with (directory / "pca_energy.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(("component", "explained_variance_pct", "cumulative_variance_pct"))
        writer.writerows(zip(components, individual, cumulative))

    fig, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.7), layout="constrained")
    shown = components <= max(30, candidates[-1])
    left.bar(components[shown], individual[shown], color="#477B9C")
    left.set(title="Individual explained variance", xlabel="Principal component", ylabel="Variance (%)")

    right.plot(components, cumulative, color="#185D68", linewidth=2)
    right.axhline(target, color="0.45", linestyle="--", linewidth=1, label=f"Target {target:g}%")
    for k in candidates:
        color, suffix = ("#B4532F", " (selected)") if k == selected else ("#185D68", "")
        right.scatter(
            k, cumulative[k - 1], color=color, s=30, zorder=3,
            label=f"K={k}: {cumulative[k - 1]:.2f}%{suffix}",
        )
    right.set(title="Cumulative explained variance", xlabel="Number of components", ylabel="Variance (%)")
    right.legend(loc="lower right", frameon=False)
    for axis in (left, right):
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("Indian Pines | PCA fitted on training spectra")
    fig.savefig(directory / "pca_energy.png", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data, labels = read_scene(args.data_dir)
    spectra = training_spectra(data, labels, args.train_ratio, args.split_seed)
    pca = PCA(svd_solver="full", whiten=True).fit(spectra)
    candidates = args.pca_components
    if candidates[-1] > pca.n_components_:
        raise ValueError(f"当前训练样本最多支持 {pca.n_components_} 个主成分，请调整 --pca-components。")

    individual = 100 * pca.explained_variance_ratio_
    if not np.isfinite(individual).all():
        raise ValueError("训练光谱的解释方差无效，请检查训练样本的光谱方差。")
    cumulative = np.cumsum(individual)
    # 完整 PCA 的累计比例为 100%，舍入误差可能让最后一项略低于目标。
    meets_target = cumulative >= args.target_variance
    if args.target_variance == 100:
        meets_target |= np.isclose(cumulative, 100, rtol=0, atol=1e-4)
    selected = next(
        (k for k in candidates if k >= MIN_HYBRIDSN_COMPONENTS and meets_target[k - 1]),
        None,
    )

    timestamp = datetime.now(SHANGHAI_TIMEZONE).strftime("%Y%m%d_%H%M%S_%f")
    directory = args.output_dir or Path(__file__).resolve().parent / f"pca_select_{timestamp}"
    directory = directory.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=False)
    export_energy(individual, cumulative, candidates, args.target_variance, selected, directory)

    print(f"训练中心像素数 {len(spectra)}（训练比例 {args.train_ratio:g}，划分种子 {args.split_seed}）")
    for k in candidates:
        print(f"K={k}  累计解释方差 {cumulative[k - 1]:.4f}%")
    print(f"初筛条件为累计解释方差 ≥ {args.target_variance:g}%，并且 K ≥ {MIN_HYBRIDSN_COMPONENTS}。")
    if selected is None:
        print("当前候选中没有满足初筛条件的维数。")
    else:
        print(f"候选维数中初选 K={selected}，最终维数结合验证集分类表现确定。")
    print(f"结果目录 {directory}")


if __name__ == "__main__":
    main()
