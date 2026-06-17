import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import os

IMAGE_FOLDER = "images"
os.makedirs(IMAGE_FOLDER, exist_ok=True)

fig_size = (1920, 1080, "px")
DISPLAY = False


def display_data(
    data: NDArray[np.float64],
    data_without_outliers: NDArray[np.float64],
    limited_outliers: NDArray[np.float64],
):
    mu, sigma = np.mean(data_without_outliers), np.std(data_without_outliers)
    x: NDArray[np.float64] = np.linspace(
        min(data_without_outliers), max(data_without_outliers), 100
    )
    y: NDArray[np.float64] = norm(x, mu, sigma)

    # Histogram with normal fit
    fullscreen()
    plt.plot(
        x,
        y,
        color="red",
        label=f"Normal fit without outliers\nμ={mu:.2f}, σ={sigma:.2f}",
    )
    plt.axvline(mu, color="black", linestyle="--")
    plt.axvline(mu - sigma, color="gray", linestyle=":")
    plt.axvline(mu + sigma, color="gray", linestyle=":")
    size = len(data_without_outliers)
    amount_of_bins = int(np.sqrt(size))
    plt.legend()
    plt.hist(
        data_without_outliers,
        density=True,
        alpha=0.6,
        color="blue",
        bins=amount_of_bins,
    )
    plt.title("Histogram without outliers")
    plt.xlabel("Values")
    plt.ylabel("Amount of values in bucket")
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_FOLDER}/histogram_without_outliers.png", dpi=300, bbox_inches="tight"
    )
    if DISPLAY:
        plt.show(block=False)

    fullscreen()

    # Chunked aggregation
    chunk_size = max(1, size // 25)
    trimmed = data_without_outliers[
        : len(data_without_outliers) // chunk_size * chunk_size
    ]
    chunks = trimmed.reshape(-1, chunk_size)
    agg = chunks.mean(axis=1)
    x_agg = np.arange(len(agg))

    plt.plot(x_agg, agg, marker="o")
    plt.title(f"Chunked aggregation (size={chunk_size})")
    plt.xlabel("Chunk index")
    plt.ylabel("Mean value")
    plt.tight_layout()
    plt.savefig(f"{IMAGE_FOLDER}/chunked_aggregation.png", dpi=300, bbox_inches="tight")
    if DISPLAY:
        plt.show(block=False)

    # Boxplot with highlighted outliers
    fullscreen()

    plt.boxplot(data, showfliers=False, showmeans=True, showcaps=True)
    plt.scatter(np.ones_like(limited_outliers), limited_outliers, color="red")
    plt.xticks([])
    plt.title("Boxplot with a couple of outliers displayed")
    plt.xlabel("")
    plt.ylabel("Values")
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_FOLDER}/boxplot_with_outliers.png", dpi=300, bbox_inches="tight"
    )
    if DISPLAY:
        plt.show(block=False)

    # Scatter plot
    fullscreen()

    plt.scatter(range(len(data_without_outliers)), data_without_outliers, s=5)
    plt.title("Scatter plot without outliers")
    plt.xlabel("Occurrence order")
    plt.xticks([])
    plt.ylabel("Values")
    plt.tight_layout()
    plt.savefig(
        f"{IMAGE_FOLDER}/scatter_without_outliers.png", dpi=300, bbox_inches="tight"
    )
    if DISPLAY:
        plt.show(block=True)


def norm(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )


def fullscreen():
    if not DISPLAY:
        plt.clf()
        plt.cla()
        plt.close()
        return

    plt.figure(figsize=fig_size)
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
