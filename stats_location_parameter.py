from typing import Any
from collections import Counter

import numpy as np
from numpy.typing import NDArray

from stat_basics import StatDetail, StatText


class LPArithmeticMean(StatDetail):
    name = StatText("Arithmetic mean", "Arithmetisches Mittel")
    short = StatText("avg", "avg")


class LPMode(StatDetail):
    name = StatText("Mode", "Modalwert")
    short = StatText("mode", "mod")
    description = StatText(
        "The value(s) that appear most frequently in the data set. There can be more than one mode if multiple values have the same highest frequency.",
        "Der Wert oder die Werte, die am häufigsten in der Datenmenge vorkommen. Es kann mehr als einen Modus geben, wenn mehrere Werte die gleiche höchste Häufigkeit haben.",
    )
    when_used = StatText(
        "Rather used for nominal or categorical data, but can be used for numerical data as well.",
        "Eher für nominale oder kategoriale Daten, aber auch für numerische Daten geeignet.",
    )


class LPPercentile(StatDetail):
    name = StatText("Percentile", "Perzentil")
    short = StatText("p", "p")
    percentile: int

    def __init__(self, value: Any, percentile: int):
        self.percentile = percentile
        super().__init__(value)

    def get_detail_string(self) -> str:
        return f"{f'{self.percentile}' + f'{self.get_name()}' + ":":<30} {self.get_short() + f'{self.percentile}':<4} = {self.value}"


class LPTrimmedMean(StatDetail):
    name = StatText("Trimmed mean", "Getrimmtes Mittel")
    short = StatText("tm", "tm")


class LPGeometricMean(StatDetail):
    name = StatText("Geometric mean", "Geometrisches Mittel")
    short = StatText("gm", "gm")


class LPHarmonicMean(StatDetail):
    name = StatText("Harmonic mean", "Harmonisches Mittel")
    short = StatText("hm", "hm")


class LPMidrange(StatDetail):
    name = StatText("Midrange", "Durchschnitt von min und max")
    short = StatText("mr", "mr")


class LPMidhinge(StatDetail):
    name = StatText("Midhinge", "Durchschnitt von q25 und q75")
    short = StatText("mh", "mh")


class LPTrimean(StatDetail):
    name = StatText("Trimean", "Dreifach-Mittel")
    short = StatText("tm", "tm")


def calculate_mode(data: NDArray[np.float64]):
    c = Counter(data)
    m = max(c.values())
    modes = [float(k) for k, v in c.items() if v == m]
    return LPMode(modes[0] if len(modes) == 1 else modes)


def calculate_trimmed_mean(data: NDArray[np.float64], proportion: float = 0.1):
    n = len(data)
    k = int(n * proportion)
    return LPTrimmedMean(np.mean(data[k : n - k]) if n > 2 * k else np.mean(data))


def calculate_geometric_mean(data: NDArray[np.float64]):
    if np.any(data <= 0):
        return LPGeometricMean(None)
    return LPGeometricMean(np.exp(np.mean(np.log(data))))


def calculate_harmonic_mean(data: NDArray[np.float64]):
    if np.any(data == 0):
        return LPHarmonicMean(None)
    return LPHarmonicMean(len(data) / np.sum(1.0 / data))


def calculate_percentile_sorted(
    sorted_data: NDArray[np.float64], p: int
) -> LPPercentile:
    """
    Nearest-rank percentile on sorted array.

    Returns the smallest value from `sorted_data` such that at least `p` percent
    of the observations are less than or equal to that value. The returned
    value is always an element of `sorted_data`.
    """
    n = len(sorted_data)
    if n == 0:
        return LPPercentile(None, p)

    # Clamp percentile to [0, 100]
    p = max(0, min(100, int(p)))

    if p == 0:
        value = sorted_data[0]
    else:
        # nearest-rank method: rank = ceil(p/100 * n)
        rank = int(np.ceil(p / 100.0 * n))
        idx = max(0, rank - 1)
        idx = min(idx, n - 1)
        value = sorted_data[idx]

    return LPPercentile(value, p)


def calculate_central_tendencies(sorted_data: NDArray[np.float64], trim: float):
    return (
        calculate_mode(sorted_data),
        calculate_geometric_mean(sorted_data),
        calculate_harmonic_mean(sorted_data),
        calculate_trimmed_mean(sorted_data, trim),
    )
