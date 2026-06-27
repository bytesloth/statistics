import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import os

from stat_constants import DISPLAY, FIG_SIZE
from stat_constants import IMAGE_FOLDER

os.makedirs(IMAGE_FOLDER, exist_ok=True)


def display_data(
    original_data: NDArray[np.float64],
    limited_outliers: NDArray[np.float64],
    data_name: str,
):
    mu, sigma = np.mean(original_data), np.std(original_data)
    x: NDArray[np.float64] = np.linspace(min(original_data), max(original_data), 100)
    y: NDArray[np.float64] = norm(x, mu, sigma)

    # Histogram with normal fit
    fullscreen()
    plt.plot(
        x,
        y,
        color="red",
        label=f"Normal fit\nμ={mu:.2f}, σ={sigma:.2f}",
    )
    plt.axvline(mu, color="black", linestyle="--")
    plt.axvline(mu - sigma, color="gray", linestyle=":")
    plt.axvline(mu + sigma, color="gray", linestyle=":")
    size = len(original_data)
    amount_of_bins = int(np.sqrt(size))
    plt.legend()
    plt.hist(
        original_data,
        density=True,
        alpha=0.6,
        color="blue",
        bins=amount_of_bins,
    )
    plt.title(f"Histogram - {data_name}")
    plt.xlabel("Values")
    plt.ylabel("Amount of values in bucket")
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_FOLDER}/histogram_{data_name}.png", dpi=300, bbox_inches="tight"
    )
    if DISPLAY:
        plt.show(block=False)

    fullscreen()

    # Chunked aggregation
    chunk_size = max(1, size // 25)
    trimmed = original_data[: len(original_data) // chunk_size * chunk_size]
    chunks = trimmed.reshape(-1, chunk_size)
    agg = chunks.mean(axis=1)
    x_agg = np.arange(len(agg))

    plt.plot(x_agg, agg, marker="o")
    plt.title(f"Chunked aggregation (size={chunk_size})")
    plt.xlabel("Chunk index")
    plt.ylabel("Mean value")
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_FOLDER}/chunked_aggregation_{data_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    if DISPLAY:
        plt.show(block=False)

    # Boxplot with highlighted outliers
    fullscreen()

    plt.boxplot(original_data, showfliers=False, showmeans=True, showcaps=True)
    plt.scatter(np.ones_like(limited_outliers), limited_outliers, color="red")
    plt.xticks([])
    plt.title(f"Boxplot - {data_name}")
    plt.xlabel("")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_FOLDER}/boxplot_{data_name}.png", dpi=300, bbox_inches="tight")
    if DISPLAY:
        plt.show(block=False)

    # Scatter plot
    fullscreen()

    plt.scatter(range(len(original_data)), original_data, s=5)
    plt.title(f"Scatter plot - {data_name}")
    plt.xlabel("Occurrence order")
    plt.xticks([])
    plt.ylabel("Values")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_FOLDER}/scatter_{data_name}.png", dpi=300, bbox_inches="tight")
    if DISPLAY:
        plt.show(block=True)


def display_data_single_figure(
    dataset_groups: dict[str, dict[str, NDArray[np.float64]]],
    image_name: str,
    title: str,
    group_by: str = "dataset",
):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    group_order = ["original", "without_outliers", "outliers"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    def get_label(dataset_name: str, group_name: str) -> str:
        if group_by == "category":
            return f"{group_name} - {dataset_name}"
        return f"{dataset_name} - {group_name}"

    def ordered_items():
        if group_by == "category":
            for group_name in group_order:
                for dataset_name, groups in dataset_groups.items():
                    yield dataset_name, group_name, groups.get(group_name)
        else:
            for dataset_name, groups in dataset_groups.items():
                for group_name in group_order:
                    yield dataset_name, group_name, groups.get(group_name)

    # Histogram
    ax = axes[0, 0]
    for dataset_name, group_name, values in ordered_items():
        if values is None or values.size == 0:
            continue
        label = get_label(dataset_name, group_name)
        color = colors[
            (
                list(dataset_groups).index(dataset_name) * len(group_order)
                + group_order.index(group_name)
            )
            % len(colors)
        ]
        ax.hist(
            values,
            bins=int(np.sqrt(values.size)) if values.size > 0 else 1,
            density=True,
            alpha=0.4,
            label=label,
            color=color,
        )

        mu = float(np.mean(values))
        sigma = float(np.std(values))
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            x_values = np.linspace(float(np.min(values)), float(np.max(values)), 100)
            y_values = norm(x_values, mu, sigma)
            ax.plot(x_values, y_values, color=color, linestyle="--", linewidth=1.5)
            ax.axvline(mu, color=color, linestyle=":", linewidth=1.0, alpha=0.8)

    ax.set_title(f"Histogram - {title}")
    ax.set_xlabel("Values")
    ax.set_ylabel("Density")
    ax.legend(fontsize="small")

    # Chunked aggregation
    ax = axes[0, 1]
    for dataset_name, group_name, values in ordered_items():
        if values is None or values.size == 0:
            continue
        chunk_size = max(1, values.size // 25)
        trimmed = values[: values.size // chunk_size * chunk_size]
        if trimmed.size == 0:
            continue
        chunks = trimmed.reshape(-1, chunk_size)
        agg = chunks.mean(axis=1)
        x_agg = np.arange(len(agg))
        ax.plot(
            x_agg,
            agg,
            marker="o",
            markersize=3,
            label=get_label(dataset_name, group_name),
            color=colors[
                (
                    list(dataset_groups).index(dataset_name) * len(group_order)
                    + group_order.index(group_name)
                )
                % len(colors)
            ],
        )
    ax.set_title(f"Chunked aggregation - {title}")
    ax.set_xlabel("Chunk index")
    ax.set_ylabel("Mean value")
    ax.legend(fontsize="small")

    # Boxplot
    ax = axes[1, 0]
    box_data = []
    labels = []
    if group_by == "category":
        for group_name in group_order:
            for dataset_name, groups in dataset_groups.items():
                values = groups.get(group_name)
                if values is None or values.size == 0:
                    continue
                box_data.append(values)
                labels.append(f"{group_name}\n{dataset_name}")
    else:
        for dataset_name, groups in dataset_groups.items():
            for group_name in group_order:
                values = groups.get(group_name)
                if values is None or values.size == 0:
                    continue
                box_data.append(values)
                labels.append(f"{dataset_name}\n{group_name}")
    if box_data:
        ax.boxplot(box_data, showfliers=False, showmeans=True, showcaps=True)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize="small")
    ax.set_title(f"Boxplot - {title}")
    ax.set_ylabel("Values")

    # Scatter plot
    ax = axes[1, 1]
    if group_by == "category":
        for group_name in group_order:
            for dataset_index, (dataset_name, groups) in enumerate(
                dataset_groups.items()
            ):
                values = groups.get(group_name)
                if values is None or values.size == 0:
                    continue
                x = np.arange(values.size) + (
                    group_order.index(group_name) + dataset_index / 10.0
                )
                ax.scatter(
                    x,
                    values,
                    s=5,
                    alpha=0.7,
                    label=f"{group_name} - {dataset_name}",
                    color=colors[
                        (
                            group_order.index(group_name) * len(dataset_groups)
                            + dataset_index
                        )
                        % len(colors)
                    ],
                )
    else:
        for dataset_index, (dataset_name, groups) in enumerate(dataset_groups.items()):
            for group_index, group_name in enumerate(group_order):
                values = groups.get(group_name)
                if values is None or values.size == 0:
                    continue
                x = np.arange(values.size) + (dataset_index + group_index / 10.0)
                ax.scatter(
                    x,
                    values,
                    s=5,
                    alpha=0.7,
                    label=f"{dataset_name} - {group_name}",
                    color=colors[
                        (dataset_index * len(group_order) + group_index) % len(colors)
                    ],
                )
    ax.set_title(f"Scatter plot - {title}")
    ax.set_xlabel("Occurrence order")
    ax.set_ylabel("Values")
    ax.legend(fontsize="small")

    fig.tight_layout()
    fig.savefig(f"{IMAGE_FOLDER}/{image_name}.png", dpi=300, bbox_inches="tight")
    if DISPLAY:
        plt.show(block=True)
    else:
        plt.close(fig)


def norm(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def fullscreen():
    if not DISPLAY:
        plt.clf()
        plt.cla()
        plt.close()
        return

    plt.figure(figsize=FIG_SIZE)
    mng = plt.get_current_fig_manager()
    # Histogram with normal fit
    try:
        # Qt backend
        if hasattr(mng, "window") and hasattr(mng.window, "showMaximized"):
            mng.window.showMaximized()
        # TkAgg backend
        elif hasattr(mng, "window") and hasattr(mng.window, "state"):
            mng.window.state("zoomed")  # Windows/Tk: maximiert mit Dekoration
        # WX backend
        elif hasattr(mng, "frame") and hasattr(mng.frame, "Maximize"):
            mng.frame.Maximize(True)
        else:
            mng.full_screen_toggle()  # Fallback: echtes Fullscreen (ohne Dekoration)
    except Exception:
        try:
            mng.full_screen_toggle()
        except Exception:
            pass
