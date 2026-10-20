from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from statistics import median
import sys
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class RunResult:
    gray: np.ndarray
    opencv: np.ndarray
    native: np.ndarray
    opencv_ms: float
    native_ms: float

    @property
    def agreement_percent(self) -> float:
        return float(np.mean(self.opencv == self.native) * 100.0)


def validate_parameters(block_size: int, c: float) -> None:
    if block_size < 3 or block_size % 2 == 0:
        raise ValueError("block_size must be an odd integer greater than or equal to 3")
    if not np.isfinite(c):
        raise ValueError("c must be a finite number")


def adaptive_threshold_native(
    gray: np.ndarray, block_size: int = 15, c: float = 7.0
) -> np.ndarray:
    validate_parameters(block_size, c)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a two-dimensional uint8 image")

    height, width = gray.shape
    radius = block_size // 2
    source = gray.tolist()

    # Повторяем крайние пиксели, чтобы даже у границ изображения
    # каждый пиксель имел полную окрестность block_size x block_size.
    padded_rows = [
        [row[0]] * radius + row + [row[-1]] * radius for row in source
    ]
    padded = (
        [padded_rows[0][:] for _ in range(radius)]
        + padded_rows
        + [padded_rows[-1][:] for _ in range(radius)]
    )

    padded_height = len(padded)
    padded_width = len(padded[0])
    # Нулевая верхняя строка и левый столбец упрощают расчёт сумм у границ.
    integral = [
        [0] * (padded_width + 1) for _ in range(padded_height + 1)
    ]

    # Строим интегральное изображение обычными циклами Python.
    # NumPy используется как формат входного и выходного изображения.
    for y, row in enumerate(padded, start=1):
        row_sum = 0
        previous = integral[y - 1]
        current = integral[y]
        for x, value in enumerate(row, start=1):
            row_sum += value
            # Сумма области сверху плюс сумма текущей строки слева до x.
            current[x] = previous[x] + row_sum

    area = block_size * block_size
    result = [[0] * width for _ in range(height)]
    for y in range(height):
        top = integral[y]
        bottom = integral[y + block_size]
        result_row = result[y]
        source_row = source[y]
        for x in range(width):
            # Находим сумму окна по четырём значениям интегрального изображения.
            # Убираем области сверху и слева, возвращаем их общее пересечение.
            local_sum = (
                bottom[x + block_size]
                - top[x + block_size]
                - bottom[x]
                + top[x]
            )
            # Вычитаем c из локального среднего, сохраняя его дробную часть.
            threshold = local_sum / area - c
            # Пиксель выше порога становится белым, остальные становятся чёрными.
            result_row[x] = 255 if source_row[x] > threshold else 0

    return np.asarray(result, dtype=np.uint8)


def adaptive_threshold_opencv(
    gray: np.ndarray, block_size: int = 15, c: float = 7.0
) -> np.ndarray:
    validate_parameters(block_size, c)
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("gray must be a two-dimensional uint8 image")
    # Готовый вариант адаптивной бинаризации по локальному среднему.
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c,
    )


def load_grayscale(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"Файл изображения не найден: {path.resolve()}\n"
            "Укажите существующий файл после --input или запустите программу "
            "без --input для демонстрационного изображения."
        )
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(
            f"Не удалось прочитать изображение: {path.resolve()}. "
            "Проверьте формат и целостность файла."
        )
    return image


def create_demo_image(width: int = 900, height: int = 520) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    background = 225 - 85 * (x / width) + 22 * np.sin(y / 75.0)
    vignette = 30 * ((x - width / 2) ** 2 + (y - height / 2) ** 2) / (
        width**2 + height**2
    )
    image = np.clip(background - vignette, 0, 255).astype(np.uint8)

    cv2.putText(
        image,
        "ADAPTIVE THRESHOLD",
        (45, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.7,
        35,
        4,
        cv2.LINE_AA,
    )
    lines = [
        "Local threshold follows illumination.",
        "Dark text remains visible on a gradient.",
        "Variant 2: adaptive mean, block = 31, C = 9.",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            image,
            line,
            (60, 210 + index * 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            45,
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(image, (560, 390), (835, 470), 55, 3)
    cv2.circle(image, (700, 430), 27, 55, 3)
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 3.0, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def run_benchmark(
    gray: np.ndarray, block_size: int, c: float, repeats: int = 5
) -> RunResult:
    opencv_times: list[float] = []
    native_times: list[float] = []
    opencv_result = native_result = None

    # Измеряем только обработку: загрузка и сохранение файлов выполняются отдельно.
    for _ in range(repeats):
        started = time.perf_counter()
        opencv_result = adaptive_threshold_opencv(gray, block_size, c)
        opencv_times.append((time.perf_counter() - started) * 1000)

        started = time.perf_counter()
        native_result = adaptive_threshold_native(gray, block_size, c)
        native_times.append((time.perf_counter() - started) * 1000)

    assert opencv_result is not None and native_result is not None
    # Медиана уменьшает влияние отдельных случайных скачков времени выполнения.
    return RunResult(
        gray=gray,
        opencv=opencv_result,
        native=native_result,
        opencv_ms=median(opencv_times),
        native_ms=median(native_times),
    )


def save_results(result: RunResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # На карте различий совпавшие пиксели чёрные, несовпавшие белые.
    difference = cv2.absdiff(result.opencv, result.native)
    cv2.imwrite(str(output_dir / "input.png"), result.gray)
    cv2.imwrite(str(output_dir / "opencv.png"), result.opencv)
    cv2.imwrite(str(output_dir / "native.png"), result.native)
    cv2.imwrite(str(output_dir / "difference.png"), difference)

    panels = []
    for title, image in (
        ("Original", result.gray),
        ("OpenCV", result.opencv),
        ("Native Python", result.native),
        ("Difference", difference),
    ):
        panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 45), (245, 245, 245), -1)
        cv2.putText(
            panel,
            title,
            (15, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (25, 25, 25),
            2,
            cv2.LINE_AA,
        )
        panels.append(panel)

    comparison = np.hstack(panels)
    max_width = 1800
    if comparison.shape[1] > max_width:
        scale = max_width / comparison.shape[1]
        comparison = cv2.resize(
            comparison,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )
    cv2.imwrite(str(output_dir / "comparison.png"), comparison)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, help="Input image. A generated demo is used if omitted."
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--block-size", type=int, default=31)
    parser.add_argument("--c", type=float, default=9.0)
    parser.add_argument("--repeats", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be at least 1")
    try:
        gray = load_grayscale(args.input) if args.input else create_demo_image()
    except (FileNotFoundError, ValueError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    result = run_benchmark(gray, args.block_size, args.c, args.repeats)
    save_results(result, args.output)

    print(f"Image size: {gray.shape[1]} x {gray.shape[0]}")
    print(f"OpenCV:       {result.opencv_ms:.3f} ms")
    print(f"Native Python: {result.native_ms:.3f} ms")
    print(f"Pixel agreement: {result.agreement_percent:.4f}%")
    print(f"Results saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
