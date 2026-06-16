from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar, Dict

LANGUAGE = "de"


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
    name: ClassVar[StatText]
    short: ClassVar[StatText]
    description: ClassVar[StatText]
    when_used: ClassVar[StatText]

    def __init__(self, value: Any):
        self.value = value

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
            #)

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

    @classmethod
    def get_name(cls, lang: str = LANGUAGE) -> str:
        return cls.name.get(lang)

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


def format_value(value: float | int | StatDetail, precision: int = 2) -> float:
    if isinstance(value, int):
        return value
    if isinstance(value, StatDetail) and isinstance(value.value, int):
        return value.value
    return round(float(value), precision)


def format_percentage(value: float | StatDetail, precision: int = 2) -> float:
    return format_value(value * 100, precision)
