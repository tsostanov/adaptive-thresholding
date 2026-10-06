import numpy as np
import pytest

from adaptive_threshold import (
    adaptive_threshold_native,
    adaptive_threshold_opencv,
    create_demo_image,
)


def test_output_is_binary_and_preserves_shape() -> None:
    image = create_demo_image(160, 100)
    result = adaptive_threshold_native(image, block_size=15, c=7)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_uniform_image_threshold() -> None:
    image = np.full((25, 31), 100, dtype=np.uint8)

    with_positive_c = adaptive_threshold_native(image, block_size=7, c=3)
    with_zero_c = adaptive_threshold_native(image, block_size=7, c=0)

    assert np.all(with_positive_c == 255)
    assert np.all(with_zero_c == 0)


def test_native_agrees_with_opencv() -> None:
    image = create_demo_image(240, 140)
    native = adaptive_threshold_native(image, block_size=31, c=9)
    opencv = adaptive_threshold_opencv(image, block_size=31, c=9)

    agreement = np.mean(native == opencv)
    assert agreement > 0.995


@pytest.mark.parametrize("block_size", [0, 2, 4, 10])
def test_invalid_block_size(block_size: int) -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError):
        adaptive_threshold_native(image, block_size=block_size)


def test_opencv_result_is_binary() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)
    result = adaptive_threshold_opencv(image, block_size=3, c=2)

    assert result.shape == image.shape
    assert result.dtype == np.uint8
    assert set(np.unique(result)).issubset({0, 255})


def test_native_matches_known_small_result() -> None:
    image = np.array(
        [
            [10, 10, 10],
            [10, 100, 10],
            [10, 10, 10],
        ],
        dtype=np.uint8,
    )

    result = adaptive_threshold_native(image, block_size=3, c=0)

    expected = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [0, 0, 0],
        ],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(result, expected)
