import argparse

import numpy as np
from numpy.typing import NDArray

from stat_basics import StatGroup, StatText, StatsTableCollector
from stats_location_parameter import (
    LPArithmeticMean,
    LPMidhinge,
    LPMidrange,
    LPPercentile,
    LPTrimean,
    calculate_central_tendencies,
    calculate_percentile_sorted,
)
from stats_data_spread import (
    DSCount,
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
    OutlierStatsCollection,
    OutlierTopCount,
    OutlierTopMean,
    OutlierTopMedian,
    calculate_outlier_stats,
)
from stats_types import calculate_integers, calculate_zero_values
from stats_visualization import display_data_single_figure

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


def describe(
    sorted_data: NDArray[np.float64],
    outlier_stats_collection: OutlierStatsCollection,
    stats_table_collector: StatsTableCollector,
    column_name: str,
    trim: float = 0.1,
):

    count = DSCount(len(sorted_data))

    # Welford pass (mean, variance, skew, kurtosis, min/max)
    w = Welford()
    for v in sorted_data:
        w.update(v)

    # central tendencies
    q1: LPPercentile = calculate_percentile_sorted(sorted_data, 1)
    q5: LPPercentile = calculate_percentile_sorted(sorted_data, 5)
    q25: LPPercentile = outlier_stats_collection.q25
    median: LPPercentile = calculate_percentile_sorted(sorted_data, 50)
    arithmetic_mean: LPArithmeticMean = LPArithmeticMean(w.mean)
    q75: LPPercentile = outlier_stats_collection.q75
    q95: LPPercentile = calculate_percentile_sorted(sorted_data, 95)
    q99: LPPercentile = calculate_percentile_sorted(sorted_data, 99)

    # Center robust metrics
    midrange = LPMidrange((w.min + w.max) / 2)
    midhinge = LPMidhinge((q25 + q75) / 2)
    trimean = LPTrimean((q25 + 2 * median + q75) / 4)

    mode, geometric_mean, harmonic_mean, contraharmonic_mean, trimmed_mean = (
        calculate_central_tendencies(sorted_data, trim)
    )

    # spread
    variance, standard_deviation, skew, kurt = w.finalize()
    inter_quantile_range = outlier_stats_collection.interquartile_range
    minimum = DSMinimum(w.min)
    maximum = DSMaximum(w.max)
    range = DSRange(w.max - w.min)
    median_absolute_deviation = calculate_median_absolute_deviation(
        sorted_data, median.value
    )
    average_absolute_deviation = calculate_average_absolute_deviation(
        sorted_data, arithmetic_mean.value
    )

    # Outlier detection
    lower_whisker = outlier_stats_collection.lower_whisker
    upper_whisker = outlier_stats_collection.upper_whisker
    bottom_outliers = sorted_data[(sorted_data < lower_whisker.value)]
    outlier_bottom_count = OutlierBottomCount(len(bottom_outliers))
    outlier_bottom_mean = OutlierBottomMean(
        np.mean(bottom_outliers) if bottom_outliers.size else np.nan
    )
    outlier_bottom_median = OutlierBottomMedian(
        np.median(bottom_outliers) if bottom_outliers.size else np.nan
    )

    top_outliers = sorted_data[(sorted_data > upper_whisker.value)]
    outlier_top_count = OutlierTopCount(len(top_outliers))
    outlier_top_mean = OutlierTopMean(
        np.mean(top_outliers) if top_outliers.size else np.nan
    )
    outlier_top_median = OutlierTopMedian(
        np.median(top_outliers) if top_outliers.size else np.nan
    )

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

    stats_table_collector.add_column(column_name)
    stats_table_collector.add_stat_row(StatGroup(StatText("Types", "Typen")))
    stats_table_collector.add_stat_row(count)
    stats_table_collector.add_stat_row(types_total_integers)
    stats_table_collector.add_stat_row(types_relative_integers)
    stats_table_collector.add_stat_row(types_total_less_than_zero)
    stats_table_collector.add_stat_row(types_relative_less_than_zero)
    stats_table_collector.add_stat_row(types_total_zero)
    stats_table_collector.add_stat_row(types_relative_zero)
    stats_table_collector.add_stat_row(types_total_more_than_zero)
    stats_table_collector.add_stat_row(types_relative_more_than_zero)

    stats_table_collector.add_stat_row(
        StatGroup(StatText("Central tendencies", "Lageparameter"))
    )
    stats_table_collector.add_stat_row(minimum)
    stats_table_collector.add_stat_row(lower_whisker)
    stats_table_collector.add_stat_row(q1)
    stats_table_collector.add_stat_row(q5)
    stats_table_collector.add_stat_row(q25)

    stats_table_collector.add_stat_row(StatGroup(StatText("Averages", "Durchschnitte")))
    stats_table_collector.add_stat_row(harmonic_mean)
    stats_table_collector.add_stat_row(geometric_mean)
    stats_table_collector.add_stat_row(trimmed_mean)
    stats_table_collector.add_stat_row(median)
    stats_table_collector.add_stat_row(trimean)
    stats_table_collector.add_stat_row(midhinge)
    stats_table_collector.add_stat_row(arithmetic_mean)
    stats_table_collector.add_stat_row(contraharmonic_mean)
    stats_table_collector.add_stat_row(midrange)
    # stats_table_collector.add_stat_row(mode)

    stats_table_collector.add_stat_row(
        StatGroup(StatText("Central tendencies2", "Lageparameter2"))
    )
    stats_table_collector.add_stat_row(q75)
    stats_table_collector.add_stat_row(q95)
    stats_table_collector.add_stat_row(q99)
    stats_table_collector.add_stat_row(upper_whisker)
    stats_table_collector.add_stat_row(maximum)

    stats_table_collector.add_stat_row(
        StatGroup(StatText("Measures of Dispersion", "Streuungsmaße"))
    )
    stats_table_collector.add_stat_row(range)
    stats_table_collector.add_stat_row(inter_quantile_range)
    stats_table_collector.add_stat_row(standard_deviation)
    stats_table_collector.add_stat_row(variance)
    stats_table_collector.add_stat_row(median_absolute_deviation)
    stats_table_collector.add_stat_row(average_absolute_deviation)
    stats_table_collector.add_stat_row(skew)
    stats_table_collector.add_stat_row(kurt)

    stats_table_collector.add_stat_row(StatGroup(StatText("Outliers", "Ausreißer")))
    stats_table_collector.add_stat_row(outlier_bottom_count)
    stats_table_collector.add_stat_row(outlier_bottom_mean)
    stats_table_collector.add_stat_row(outlier_bottom_median)
    stats_table_collector.add_stat_row(outlier_top_count)
    stats_table_collector.add_stat_row(outlier_top_mean)
    stats_table_collector.add_stat_row(outlier_top_median)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Statistics collector and visualizer for multiple datasets."
    )
    parser.add_argument(
        "--group-by",
        choices=["separate", "dataset", "category", "all"],
        default="category",
        help=(
            "separate: one image for each dataset/category combination; "
            "dataset: one image per dataset with all categories combined; "
            "category: one figure per category comparing all datasets within that category; "
            "all: one image with all datasets and all categories combined;"
        ),
    )
    parser.add_argument("file_names", nargs="+", help="Input files")
    return parser.parse_args()


def main():
    args = parse_args()
    group_by = args.group_by
    file_names = args.file_names

    stats_table_collector = StatsTableCollector()
    dataset_groups: dict[str, dict[str, np.ndarray]] = {}

    for file_name in file_names:
        data = np.asarray(np.loadtxt(file_name, delimiter=","), dtype=np.float64)
        original_data = np.asarray(data, dtype=np.float64)
        if original_data.size == 0:
            raise ValueError("Empty dataset")

        # Sort once (needed for percentiles + median + order stats)
        sorted_data = np.sort(original_data)
        original_outlier_stats_collection = process_data(
            sorted_data, stats_table_collector, f"Orig_{file_name}"
        )

        data_without_outliers = sorted_data[
            (original_outlier_stats_collection.lower_whisker.value < sorted_data)
            & (sorted_data < original_outlier_stats_collection.upper_whisker.value)
        ]
        process_data(
            data_without_outliers,
            stats_table_collector,
            f"-Aus_{file_name}",
        )

        outliers = sorted_data[
            (sorted_data < original_outlier_stats_collection.lower_whisker.value)
            | (original_outlier_stats_collection.upper_whisker.value < sorted_data)
        ]
        process_data(outliers, stats_table_collector, f"Aus_{file_name}")

        dataset_label = file_name.replace(".txt", "")
        original_data_without_outliers = original_data[
            (original_outlier_stats_collection.lower_whisker.value < original_data)
            & (original_data < original_outlier_stats_collection.upper_whisker.value)
        ]
        original_outliers = original_data[
            (original_data < original_outlier_stats_collection.lower_whisker.value)
            | (original_outlier_stats_collection.upper_whisker.value < original_data)
        ]
        dataset_groups[dataset_label] = {
            "original": original_data,
            "without_outliers": original_data_without_outliers,
            "outliers": original_outliers,
        }

    if group_by == "category" and dataset_groups:
        for category in ["original", "without_outliers", "outliers"]:
            display_data_single_figure(
                {
                    dataset_label: {category: dataset_groups[dataset_label][category]}
                    for dataset_label in dataset_groups
                },
                image_name=f"summary_{category}",
                title=f"{category} comparison",
                group_by="dataset",
            )

    if group_by == "separate" and dataset_groups:
        for dataset_label in dataset_groups:
            for category in ["original", "without_outliers", "outliers"]:
                display_data_single_figure(
                    {
                        dataset_label: {
                            category: dataset_groups[dataset_label][category]
                        }
                    },
                    image_name=f"summary_{dataset_label}_{category}",
                    title=f"{dataset_label} - {category}",
                    group_by="dataset",
                )

    if group_by == "dataset" and dataset_groups:
        for dataset_label in dataset_groups:
            display_data_single_figure(
                {dataset_label: dataset_groups[dataset_label]},
                image_name=f"summary_{dataset_label}",
                title=f"{dataset_label}",
                group_by="category",
            )

    if group_by == "all" and dataset_groups:
        display_data_single_figure(
            dataset_groups,
            image_name="summary_all_datasets",
            title="All datasets",
            group_by="category",
        )

    stats_table_collector.print_table()


def process_data(
    sorted_data: np.ndarray,
    stats_table_collector: StatsTableCollector,
    data_name: str,
):
    outlier_stats_collection = calculate_outlier_stats(sorted_data)
    describe(
        sorted_data,
        outlier_stats_collection,
        stats_table_collector,
        data_name,
    )

    return outlier_stats_collection


if __name__ == "__main__":
    main()
