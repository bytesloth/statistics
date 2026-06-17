import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def make_scatter(
    data: NDArray[np.float64],
    data_without_outliers: NDArray[np.float64],
    limited_outliers: NDArray[np.float64],
):

    mu, sigma = np.mean(data_without_outliers), np.std(data_without_outliers)
    x = np.linspace(min(data_without_outliers), max(data_without_outliers), 100)
    y = norm(x, mu, sigma)
    plt.plot(x, y, color="red", label=f"Normal fit without outliers\nμ={mu:.2f}, σ={sigma:.2f}")

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
        bins=amount_of_bins
    )
    plt.title("Histogram without outliers")
    plt.xlabel("Values")
    plt.ylabel("Amount of values in bucket")
    plt.show()


    chunk_size = size // 25
    trimmed = data_without_outliers[:len(data_without_outliers) // chunk_size * chunk_size]
    chunks = trimmed.reshape(-1, chunk_size)
    agg = chunks.mean(axis=1)
    x = np.arange(len(agg))
    plt.plot(x, agg, marker="o")
    plt.title("Chunked aggregation (size=10)")
    plt.xlabel("Chunk index")
    plt.ylabel("Mean value")
    plt.show()



    plt.boxplot(data, showfliers=False, showmeans=True, showcaps=True)
    plt.scatter(np.ones_like(limited_outliers), limited_outliers, color="red")
    plt.xticks([])
    plt.title("Boxplot with a couple of outliers displayed")
    plt.xlabel("")
    plt.ylabel("Values")
    plt.show()

    plt.scatter(range(len(data_without_outliers)), data_without_outliers, s=5)
    plt.title("Scatter plot without outliers")
    plt.xlabel("Occurrence order")
    plt.xticks([])
    plt.ylabel("Values")
    plt.show()
    # plt.savefig()


def norm(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(
        -((x - mu) ** 2) / (2 * sigma**2)
    )
