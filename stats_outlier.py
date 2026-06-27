import numpy as np

from stat_basics import StatDetail, StatText
from stats_data_spread import DSInterQuantileRange
from stats_location_parameter import LPPercentile, calculate_percentile_sorted


class OutlierBottomCount(StatDetail):
    name = StatText("Bottom Outlier Count", "Anzahl unterer Ausreißer")
    short = StatText("bon")


class OutlierBottomMean(StatDetail):
    name = StatText("Bottom Outlier Mean", "Avg der unteren Ausreißer")
    short = StatText("oavg")


class OutlierBottomMedian(StatDetail):
    name = StatText("Bottom Outlier Median", "Median der unteren Ausreißer")
    short = StatText("obm")


class OutlierTopCount(StatDetail):
    name = StatText("Top Outlier Count", "Anzahl oberer Ausreißer")
    short = StatText("ton")


class OutlierTopMean(StatDetail):
    name = StatText("Top Outlier Mean", "Avg der oberen Ausreißer")
    short = StatText("oavg")


class OutlierTopMedian(StatDetail):
    name = StatText("Top Outlier Median", "Median der oberen Ausreißer")
    short = StatText("otm")


class OutlierLowerWhisker(StatDetail):
    name = StatText("Lower Whisker", "Untere Whisker")
    short = StatText("lw")


class OutlierUpperWhisker(StatDetail):
    name = StatText("Upper Whisker", "Obere Whisker")
    short = StatText("uw")


class OutlierStatsCollection:
    def __init__(
        self,
        q25: LPPercentile,
        q75: LPPercentile,
        interquartile_range: DSInterQuantileRange,
        lower_whisker: OutlierLowerWhisker,
        upper_whisker: OutlierUpperWhisker,
    ):
        self.q25: LPPercentile = q25
        self.q75: LPPercentile = q75
        self.interquartile_range: DSInterQuantileRange = interquartile_range
        self.lower_whisker: OutlierLowerWhisker = lower_whisker
        self.upper_whisker: OutlierUpperWhisker = upper_whisker


def calculate_outlier_stats(sorted_data: np.ndarray) -> OutlierStatsCollection:
    q25: LPPercentile = calculate_percentile_sorted(sorted_data, 25)
    q75: LPPercentile = calculate_percentile_sorted(sorted_data, 75)
    inter_quantile_range: DSInterQuantileRange = DSInterQuantileRange(q75 - q25)
    lower_whisker = OutlierLowerWhisker(q25 - 1.5 * inter_quantile_range)
    upper_whisker = OutlierUpperWhisker(q75 + 1.5 * inter_quantile_range)
    return OutlierStatsCollection(
        q25=q25,
        q75=q75,
        interquartile_range=inter_quantile_range,
        lower_whisker=lower_whisker,
        upper_whisker=upper_whisker,
    )
