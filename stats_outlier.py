from stat_basics import StatDetail, StatText




class OutlierBottomCount(StatDetail):
    name = StatText("Bottom Outlier Count", "Anzahl unterer Ausreißer")
    short = StatText("boutn")


class OutlierBottomMean(StatDetail):
    name = StatText("Outlier Mean", "Mittelwert der Ausreißer")
    short = StatText("oavg")

class OutlierBottomMedian(StatDetail):
    name = StatText("Outlier Median", "Median der Ausreißer")
    short = StatText("omed")

class OutlierTopCount(StatDetail):
    name = StatText("Top Outlier Count", "Anzahl oberer Ausreißer")
    short = StatText("toutn")


class OutlierTopMean(StatDetail):
    name = StatText("Outlier Mean", "Mittelwert der Ausreißer")
    short = StatText("oavg")


class OutlierTopMedian(StatDetail):
    name = StatText("Outlier Median", "Median der Ausreißer")
    short = StatText("omed")


class OutlierLowerWhisker(StatDetail):
    name = StatText("Lower Whisker", "Untere Whisker")
    short = StatText("lw")


class OutlierUpperWhisker(StatDetail):
    name = StatText("Upper Whisker", "Obere Whisker")
    short = StatText("uw")
