from typing import List

from twinkle.trajectory import Trajectory


class DataProcessor:

    def __call__(self, rows) -> List[Trajectory]:
        ...


class DataFilter:

    def __call__(self, row) -> bool:
        ...