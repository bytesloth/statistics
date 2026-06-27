from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Dict

from stat_constants import LANGUAGE


class StatsTableCollector:
    """Collect statistics rows and render them as a simple table."""

    def __init__(self):
        self.columns: list[str] = []
        self.current_column: str | None = None
        self.row_order: list[str] = []
        self.rows: dict[str, dict[str, StatDetail | None]] = {}
        self.row_labels: dict[str, str] = {}
        self.row_types: dict[str, str] = {}
        self.stats_by_type: dict[type, list[StatDetail]] = {}

    def add_column(self, name: str) -> None:
        if name not in self.columns:
            self.columns.append(name)
        self.current_column = name
        for row_values in self.rows.values():
            row_values.setdefault(name, None)

    def _get_row_key(self, stat: "StatDetail") -> str:
        if isinstance(stat, StatGroup):
            return f"__group__{stat.get_name()}"

        percentile = (
            stat.__dict__.get("percentile") if hasattr(stat, "__dict__") else None
        )
        if isinstance(percentile, int):
            return f"{type(stat).__name__}:{percentile}"
        return type(stat).__name__

    def _get_row_label(self, stat: "StatDetail") -> str:
        percentile = (
            stat.__dict__.get("percentile") if hasattr(stat, "__dict__") else None
        )
        if isinstance(percentile, int):
            return f"p{percentile}"
        return stat.get_name()

    def add_row(self, label: str, row_key: str | None = None) -> str:
        key = row_key or label
        if key not in self.rows:
            self.row_order.append(key)
            self.rows[key] = {column: None for column in self.columns}
            self.row_labels[key] = label
            self.row_types[key] = "stat"
        return key

    def add_group_row(self, label: str, row_key: str | None = None) -> str:
        key = row_key or f"__group__{label}"
        if key not in self.rows:
            self.row_order.append(key)
            self.rows[key] = {column: None for column in self.columns}
            self.row_labels[key] = label
            self.row_types[key] = "group"
        return key

    def add_stat_row(self, stat: StatDetail) -> None:
        column_name: str = self.current_column

        self.add_column(column_name)
        row_key = self._get_row_key(stat)
        self.add_row(self._get_row_label(stat), row_key)

        if isinstance(stat, StatGroup):
            self.row_types[row_key] = "group"
        else:
            self.rows[row_key][column_name] = stat
            stat_type = type(stat)
            if stat_type not in self.stats_by_type:
                self.stats_by_type[stat_type] = []
            self.stats_by_type[stat_type].append(stat)

    def get_row_values(self, row_key: str) -> list[StatDetail | None]:
        return [self.rows[row_key][column] for column in self.columns]

    def _format_value(self, stat: StatDetail | None) -> str:
        if stat is None:
            return ""
        value = getattr(stat, "value", stat)
        if value is None:
            return "None"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def _format_abbreviation(self, stat: StatDetail | None) -> str:
        if stat is None:
            return ""
        return f"{stat.get_short()} ="

    def _format_cell(self, stat: StatDetail | None) -> str:
        return self._format_value(stat)

    def _is_group_row(self, row_key: str) -> bool:
        return self.row_types.get(row_key) == "group"

    def render_table(self) -> str:
        if not self.columns:
            return ""

        headers = ["Statistic", *self.columns]
        widths = [len(header) for header in headers]
        rows: list[list[str]] = []

        for row_key in self.row_order:
            if self._is_group_row(row_key):
                row_values = [self.row_labels[row_key]] + ["" for _ in self.columns]
            else:
                first_stat = (
                    self.rows[row_key].get(self.columns[0]) if self.columns else None
                )
                label = self.row_labels[row_key]
                abbreviation = (
                    self._format_abbreviation(first_stat)
                    if first_stat is not None
                    else ""
                )

                if abbreviation:
                    separator = ": "
                    label_prefix = f"{label}{separator}"
                    first_cell = f"{label_prefix}{abbreviation}"
                else:
                    first_cell = label

                row_values = [first_cell] + [
                    self._format_cell(self.rows[row_key].get(column))
                    for column in self.columns
                ]

            rows.append(row_values)
            for index, cell in enumerate(row_values):
                widths[index] = max(widths[index], len(cell))

        lines: list[str] = []
        header_line = " | ".join(
            header.ljust(widths[index]) for index, header in enumerate(headers)
        )
        lines.append(header_line)
        lines.append("-+-".join("-" * width for width in widths))

        for row_key, cells in zip(self.row_order, rows):
            if self._is_group_row(row_key):
                if lines:
                    lines.append("")
                first_cell = cells[0].ljust(widths[0])
                formatted_cells = [first_cell] + [
                    cell.ljust(widths[index])
                    for index, cell in enumerate(cells[1:], start=1)
                ]
                lines.append(" | ".join(formatted_cells))
                continue

            first_stat = (
                self.rows[row_key].get(self.columns[0]) if self.columns else None
            )
            label = self.row_labels[row_key]
            abbreviation = (
                self._format_abbreviation(first_stat) if first_stat is not None else ""
            )

            if abbreviation:
                separator = ": "
                label_prefix = f"{label}{separator}"
                padded_abbreviation = abbreviation.rjust(widths[0] - len(label_prefix))
                first_cell = f"{label_prefix}{padded_abbreviation}"
            else:
                first_cell = label.ljust(widths[0])

            formatted_cells = [first_cell] + [
                cell.ljust(widths[index])
                for index, cell in enumerate(cells[1:], start=1)
            ]
            lines.append(" | ".join(formatted_cells))

        return "\n".join(lines)

    def print_table(self) -> None:
        print(self.render_table())


@dataclass(frozen=True)
class StatText:
    en: str
    de: str | None = None

    def __post_init__(self) -> None:
        if self.de is None:
            object.__setattr__(self, "de", self.en)

    def get(self, lang: str = "en") -> str:
        if lang == "de":
            return self.de
        return self.en

    def as_dict(self) -> Dict[str, str]:
        return {
            "en": self.en,
            "de": self.de,
        }


class StatDetail(ABC):
    value: Any
    name: StatText
    short: ClassVar[StatText]
    description: ClassVar[StatText]
    when_used: ClassVar[StatText]

    def __init__(self, value: Any, name: StatText|None=None):
        self.value = value
        if name:
            self.name = name

    def __init_subclass__(cls, **kwargs: Any):
        super().__init_subclass__(**kwargs)
        if cls is StatDetail:
            return

        required = ["name", "short", "description", "when_used"]
        missing = [
            attr
            for attr in required
            if not isinstance(getattr(cls, attr, None), StatText)
        ]
        if missing:
            pass
        # raise TypeError(
        #     f"{cls.__name__} must define class attributes: {', '.join(missing)}"
        # )

    def __getattr__(self, _) -> Any:
        return self.value

    def __str__(self) -> str:
        return str(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __mul__(self, other: Any) -> Any:
        return self.value * other

    def __rmul__(self, other: Any) -> Any:
        return other * self.value

    def __truediv__(self, other: Any) -> Any:
        return self.value / other

    def __rtruediv__(self, other: Any) -> Any:
        return other / self.value

    def __add__(self, other: Any) -> Any:
        return self.value + other

    def __radd__(self, other: Any) -> Any:
        return other + self.value

    def __sub__(self, other: Any) -> Any:
        return self.value - other

    def __rsub__(self, other: Any) -> Any:
        return other - self.value

    def __lt__(self, other: Any) -> bool:
        return self.value < other

    def __gt__(self, other: Any) -> bool:
        return self.value > other

    def get_name(self, lang: str = LANGUAGE) -> str:
        return self.name.get(lang)

    @classmethod
    def get_description(cls, lang: str = LANGUAGE) -> str:
        return cls.description.get(lang)

    @classmethod
    def get_when_used(cls, lang: str = LANGUAGE) -> str:
        return cls.when_used.get(lang)

    @classmethod
    def get_short(cls, lang: str = LANGUAGE) -> str:
        return cls.short.get(lang)

    def get_detail_string(self) -> str:
        return f"{self.get_name() + ":":<30} {self.get_short():<4} = {self.value}"

    def print_detail(self) -> None:
        print(self.get_detail_string())


class StatGroup(StatDetail):
    def __init__(self, stat_text: StatText):
        super().__init__(None, stat_text)

    def get_name(self, lang: str = LANGUAGE) -> str:
        return self.name.get(lang)

    @classmethod
    def get_short(cls, lang: str = LANGUAGE) -> str:
        return ""

    def get_detail_string(self) -> str:
        return self.get_name()

    def print_detail(self) -> None:
        print(self.get_name())


def format_value(value: float | int | StatDetail, precision: int = 2) -> float:
    if isinstance(value, int):
        return value
    if isinstance(value, StatDetail) and isinstance(value.value, int):
        return value.value
    return round(float(value), precision)


def format_percentage(value: float | StatDetail, precision: int = 2) -> float:
    return format_value(value * 100, precision)
