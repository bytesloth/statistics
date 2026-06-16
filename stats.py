from typing import Any

import numpy as np

from stat_basics import format_percentage
from stats_location_parameter import (
    LPArithmeticMean,
    LPMidhinge,
    LPMidrange,
    LPTrimean,
    calculate_central_tendencies,
    calculate_percentile_sorted,
)
from stats_data_spread import (
    DSCount,
    DSInterQuantileRange,
    DSKurtosis,
    DSMaximum,
    DSMinimum,
    DSRange,
    DSSkewness,
    DSStandardDeviation,
    DSVariance,
    calculate_average_absolute_deviation,
    calculate_median_absolute_deviation,
)
from stats_outlier import (
    OutlierBottomCount,
    OutlierBottomMean,
    OutlierBottomMedian,
    OutlierLowerWhisker,
    OutlierTopCount,
    OutlierTopMean,
    OutlierTopMedian,
    OutlierUpperWhisker,
)
from stats_types import calculate_integers, calculate_zero_values

# ----------------------------
# Core single-pass accumulator
# ----------------------------


class Welford:
    """
    Online algorithm for mean, variance, skewness, kurtosis.
    Numerically stable and O(n).
    """

    def __init__(self):
        self.n: int = 0
        self.mean: float = 0.0
        self.m2 = 0.0
        self.m3 = 0.0
        self.m4 = 0.0

        self.min = float("inf")
        self.max = float("-inf")

    def update(self, x: float):
        n1 = self.n
        self.n += 1
        delta = x - self.mean
        delta_n = delta / self.n
        delta_n2 = delta_n * delta_n
        term1 = delta * delta_n * n1

        self.mean += delta_n

        self.m4 += (
            term1 * delta_n2 * (self.n * self.n - 3 * self.n + 3)
            + 6 * delta_n2 * self.m2
            - 4 * delta_n * self.m3
        )

        self.m3 += term1 * delta_n * (self.n - 2) - 3 * delta_n * self.m2

        self.m2 += term1

        self.min = min(self.min, x)
        self.max = max(self.max, x)

    def finalize(self):
        if self.n < 2:
            return DSVariance(0), DSStandardDeviation(0), DSSkewness(0), DSKurtosis(0)

        variance = self.m2 / self.n
        std = np.sqrt(variance)

        skew = (np.sqrt(self.n) * self.m3) / (self.m2**1.5) if self.m2 else 0
        kurt = (self.n * self.m4) / (self.m2 * self.m2) - 3 if self.m2 else 0

        return (
            DSVariance(variance),
            DSStandardDeviation(std),
            DSSkewness(skew),
            DSKurtosis(kurt),
        )


def describe(data: list[float], trim: float = 0.1):
    x = np.asarray(data, dtype=np.float64)
    if x.size == 0:
        raise ValueError("Empty dataset")

    # Sort once (needed for percentiles + median + order stats)
    sorted_data = np.sort(x)

    # Welford pass (mean, variance, skew, kurtosis, min/max)
    w = Welford()
    for v in sorted_data:
        w.update(v)

    # central tendencies
    q1 = calculate_percentile_sorted(sorted_data, 1)
    q5 = calculate_percentile_sorted(sorted_data, 5)
    q25 = calculate_percentile_sorted(sorted_data, 25)
    median = calculate_percentile_sorted(sorted_data, 50)
    arithmetic_mean = LPArithmeticMean(w.mean)
    q75 = calculate_percentile_sorted(sorted_data, 75)
    q95 = calculate_percentile_sorted(sorted_data, 95)
    q99 = calculate_percentile_sorted(sorted_data, 99)

    # Center robust metrics
    midrange = LPMidrange((w.min + w.max) / 2)
    midhinge = LPMidhinge((q25 + q75) / 2)
    trimean = LPTrimean((q25 + 2 * median + q75) / 4)

    mode, geometric_mean, harmonic_mean, trimmed_mean = calculate_central_tendencies(
        sorted_data, trim
    )

    # spread
    count = DSCount(len(sorted_data))
    variance, standard_deviation, skew, kurt = w.finalize()
    inter_quantile_range = DSInterQuantileRange(q75 - q25)
    minimum = DSMinimum(w.min)
    maximum = DSMaximum(w.max)
    range = DSRange(w.max - w.min)
    median_absolute_deviation = calculate_median_absolute_deviation(sorted_data, median)
    average_absolute_deviation = calculate_average_absolute_deviation(
        sorted_data, arithmetic_mean
    )

    # Outlier detection
    lower_whisker = OutlierLowerWhisker(q25 - 1.5 * inter_quantile_range)
    upper_whisker = OutlierUpperWhisker(q75 + 1.5 * inter_quantile_range)
    bottom_outliers = sorted_data[(sorted_data < lower_whisker.value)]
    outlier_bottom_count = OutlierBottomCount(len(bottom_outliers))
    outlier_bottom_mean = OutlierBottomMean(np.mean(bottom_outliers))
    outlier_bottom_median = OutlierBottomMedian(np.median(bottom_outliers))

    top_outliers = sorted_data[(sorted_data > upper_whisker.value)]
    outlier_top_count = OutlierTopCount(len(top_outliers))
    outlier_top_mean = OutlierTopMean(np.mean(top_outliers))
    outlier_top_median = OutlierTopMedian(np.median(top_outliers))

    # types
    types_total_integers, types_relative_integers = calculate_integers(sorted_data)
    (
        types_total_less_than_zero,
        types_relative_less_than_zero,
        types_total_zero,
        types_relative_zero,
        types_total_more_than_zero,
        types_relative_more_than_zero,
    ) = calculate_zero_values(sorted_data)

    print_stats(f"{count.get_detail_string()}")
    print_stats(
        f"{types_total_integers.get_detail_string()} ({format_percentage(types_relative_integers)}%)"
    )
    print_stats(
        f"{types_total_less_than_zero.get_detail_string()} ({format_percentage(types_relative_less_than_zero)}%)"
    )
    print_stats(
        f"{types_total_zero.get_detail_string()} ({format_percentage(types_relative_zero)}%)"
    )
    print_stats(
        f"{types_total_more_than_zero.get_detail_string()} ({format_percentage(types_relative_more_than_zero)}%)"
    )
    print_stats("")
    print_stats(f"{minimum.get_detail_string()}")
    print_stats(f"{lower_whisker.get_detail_string()}")
    print_stats(f"{q1.get_detail_string()}")
    print_stats(f"{q5.get_detail_string()}")
    print_stats(f"{q25.get_detail_string()}")
    print_stats(f"{median.get_detail_string()}")

    print_stats(f"{arithmetic_mean.get_detail_string()}")
    print_stats(f"{trimmed_mean.get_detail_string()}")
    print_stats(f"{midrange.get_detail_string()}")
    print_stats(f"{midhinge.get_detail_string()}")
    print_stats(f"{trimean.get_detail_string()}")
    print_stats(f"{mode.get_detail_string()}")
    print_stats(f"{geometric_mean.get_detail_string()}")
    print_stats(f"{harmonic_mean.get_detail_string()}")

    print_stats(f"{q75.get_detail_string()}")
    print_stats(f"{q95.get_detail_string()}")
    print_stats(f"{q99.get_detail_string()}")
    print_stats(f"{upper_whisker.get_detail_string()}")
    print_stats(f"{maximum.get_detail_string()}")

    print_stats("")
    print_stats(f"Bottom Outliers: {bottom_outliers}")
    print_stats(f"{outlier_bottom_count.get_detail_string()}")
    print_stats(f"{outlier_bottom_mean.get_detail_string()}")
    print_stats(f"{outlier_bottom_median.get_detail_string()}")

    print_stats(f"Top Outliers: {top_outliers}")
    print_stats(f"{outlier_top_count.get_detail_string()}")
    print_stats(f"{outlier_top_mean.get_detail_string()}")
    print_stats(f"{outlier_top_median.get_detail_string()}")

    print_stats("")
    print_stats(f"{range.get_detail_string()}")
    print_stats(f"{inter_quantile_range.get_detail_string()}")
    print_stats(f"{standard_deviation.get_detail_string()}")
    print_stats(f"{variance.get_detail_string()}")
    print_stats(f"{median_absolute_deviation.get_detail_string()}")
    print_stats(f"{average_absolute_deviation.get_detail_string()}")
    # TODO
    print_stats(
        f"{skew.get_detail_string()} ({('right tail is longer' if skew > 0 else 'left tail is longer')})"
    )
    print_stats(
        f"{kurt.get_detail_string()} ({('steilgipflig' if kurt > 0 else 'flachgipflig')})"
    )


def print_stats(data: Any):
    print(data)


if __name__ == "__main__":
    data: list[float] = [1, 5, 7, -5, 5.11245, 100, -54, 0, 1.0, -10.01, 0]
    describe(data)
