import numpy as np
from numpy.typing import NDArray

from stat_basics import StatDetail, StatText


class DSCount(StatDetail):
    name = StatText("Count", "Anzahl")
    short = StatText("n")


class DSInterQuantileRange(StatDetail):
    name = StatText("Inter-Quantile Range", "Interquartilsabstand")
    short = StatText("iqr", "iqr")


class DSMinimum(StatDetail):
    name = StatText("Minimum", "Minimum")
    short = StatText("min")


class DSMaximum(StatDetail):
    name = StatText("Maximum", "Maximum")
    short = StatText("max")


class DSRange(StatDetail):
    name = StatText("Range", "Spannweite")
    short = StatText("rng", "spw")


class DSVariance(StatDetail):
    name = StatText("Variance", "Varianz")
    short = StatText("var")


class DSStandardDeviation(StatDetail):
    name = StatText("Standard Deviation", "Standardabweichung")
    short = StatText("std")


class DSMedianAbsoluteDeviation(StatDetail):
    name = StatText("Median Absolute Deviation", "Mediane Absolute Abweichung")
    short = StatText("mad", "mad")


class DSAverageAbsoluteDeviation(StatDetail):
    name = StatText("Average Absolute Deviation", "Mittlere Absolute Abweichung")
    short = StatText("aad", "md")


# distributions???


class DSSkewness(StatDetail):
    name = StatText("Skewness", "Schiefe")
    short = StatText("skew")
    description = StatText(
        "Refers to the degree of asymmetry in the probability distribution",
        "Maß für die Asymmetrie einer (eingipfligen) Wahrscheinlichkeitsfunktion, statistischen Dichtefunktion",
    )


class DSKurtosis(StatDetail):
    name = StatText("Kurtosis", "Wölbung")
    short = StatText("kurt")
    description = StatText(
        "Refers to the degree of tailedness in the probability distribution",
        'Steilheit bzw. "Spitzigkeit" einer (eingipfligen) Wahrscheinlichkeitsfunktion, statistischen Dichtefunktion',
    )


def calculate_median_absolute_deviation(data: NDArray[np.float64], median: float):
    return DSMedianAbsoluteDeviation(np.median(np.abs(data - median)))


def calculate_average_absolute_deviation(data: NDArray[np.float64], mean: float):
    return DSAverageAbsoluteDeviation(np.mean(np.abs(data - mean)))
