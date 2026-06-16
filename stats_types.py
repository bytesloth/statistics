import numpy as np
from numpy.typing import NDArray

from stat_basics import StatDetail, StatText


class TypesTotalIntegers(StatDetail):
    name = StatText("Total integers", "Gesamt ganze Zahlen")
    short = StatText("int")


class TypesRelativeIntegers(StatDetail):
    name = StatText("Relative integers", "Relative ganze Zahlen")
    short = StatText("int%")


class TypesTotalLessThanZero(StatDetail):
    name = StatText("Total less than zero", "Gesamt weniger als null")
    short = StatText("<0")


class TypesRelativeLessThanZero(StatDetail):
    name = StatText("Relative less than zero", "Relativ weniger als null")
    short = StatText("<0%")


class TypesTotalZero(StatDetail):
    name = StatText("Total zero", "Gesamt gleich null")
    short = StatText("=0")


class TypesRelativeZero(StatDetail):
    name = StatText("Relative zero", "Relativ gleich null")
    short = StatText("=0%")


class TypesTotalMoreThanZero(StatDetail):
    name = StatText("Total more than zero", "Gesamt mehr als null")
    short = StatText(">0")


class TypesRelativeMoreThanZero(StatDetail):
    name = StatText("Relative more than zero", "Relativ mehr als null")
    short = StatText(">0%")


def calculate_integers(sorted_data: NDArray[np.float64]):
    count = int(np.sum(np.isclose(sorted_data, np.round(sorted_data))))
    return TypesTotalIntegers(count), TypesRelativeIntegers(count / sorted_data.size)


def calculate_zero_values(sorted_data: NDArray[np.float64]):
    count_less_than_zero = 0
    count_zero = 0
    count_more_than_zero = 0
    for date in sorted_data:
        if date < 0:
            count_less_than_zero += 1
        elif date == 0:
            count_zero += 1
        else:
            count_more_than_zero += 1
    size = sorted_data.size
    return (
        TypesTotalLessThanZero(count_less_than_zero),
        TypesRelativeLessThanZero(count_less_than_zero / size),
        TypesTotalZero(count_zero),
        TypesRelativeZero(count_zero / size),
        TypesTotalMoreThanZero(count_more_than_zero),
        TypesRelativeMoreThanZero(count_more_than_zero / size),
    )
