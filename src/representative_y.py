from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
import numpy as np
import numpy.typing as npt


_NEIGHBOR_OFFSETS = [
    (0, 1),  # 上
    (-1, 0),  # 左
    (1, 0),  # 右
]


def walk_connected_pixels(
    mask: npt.NDArray[np.bool_],
    tip_x: int,
    tip_y: int,
):
    height, width = mask.shape

    if not (0 <= tip_x < width and 0 <= tip_y < height):
        raise ValueError(f'({tip_x=}, {tip_y=}) is out of bounds')

    if not mask[tip_y, tip_x]:
        raise ValueError(f'({tip_x=}, {tip_y=}) is not on the mask')

    visited = np.zeros_like(mask, dtype=bool)
    queue = deque([(tip_x, tip_y)])
    visited[tip_y, tip_x] = True

    while queue:
        x, y = queue.popleft()
        yield x, y

        for dx, dy in _NEIGHBOR_OFFSETS:
            nx = x + dx
            ny = y + dy

            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if visited[ny, nx]:
                continue
            if not mask[ny, nx]:
                continue

            visited[ny, nx] = True
            queue.append((nx, ny))


class WidthEstimator(ABC):
    @abstractmethod
    def estimate(self, mask: npt.NDArray[np.bool_], x: int, y: int) -> int:
        pass


@dataclass
class HorizontalRunLengthEstimator(WidthEstimator):
    _width_cache_by_row: dict[int, npt.NDArray[np.int_]] = field(
        default_factory=dict, init=False
    )

    def estimate(self, mask: npt.NDArray[np.bool_], x: int, y: int) -> int:
        if not mask[y, x]:
            return 0

        row_cache = self._width_cache_by_row.get(y)
        if row_cache is None:
            row_cache = np.full(mask.shape[1], -1, dtype=np.int_)
            self._width_cache_by_row[y] = row_cache

        cached_width = row_cache[x]
        if cached_width >= 0:
            return cached_width

        left = x
        while left - 1 >= 0 and mask[y, left - 1]:
            left -= 1

        right = x
        row_width = mask.shape[1]
        while right + 1 < row_width and mask[y, right + 1]:
            right += 1

        run_width = right - left + 1
        row_cache[left : right + 1] = run_width
        return run_width


class PeakRepresentativeYStrategy(ABC):
    @abstractmethod
    def compute_representative_y(
        self, mask: npt.NDArray[np.bool_], tip_x: int, tip_y: int
    ) -> int:
        pass


@dataclass(frozen=True)
class FirstPointMeetingWidthFromTipStrategy(PeakRepresentativeYStrategy):
    width_estimator: WidthEstimator
    min_width: int = 2

    def compute_representative_y(
        self, mask: npt.NDArray[np.bool_], tip_x: int, tip_y: int
    ) -> int:
        for x, y in walk_connected_pixels(mask=mask, tip_x=tip_x, tip_y=tip_y):
            width = self.width_estimator.estimate(mask, x, y)
            if width >= self.min_width:
                return y

        return mask.shape[0]
