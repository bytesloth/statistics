from stat_basics import StatDetail, StatText


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
