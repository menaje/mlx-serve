"""Image router runtime-directory regression tests."""

from mlx_serve.routers import images


def test_cleanup_recreates_missing_temp_directory(tmp_path, monkeypatch):
    image_dir = tmp_path / "missing-images"
    monkeypatch.setattr(images, "_TEMP_IMAGE_DIR", image_dir)

    images._cleanup_old_images()

    assert image_dir.is_dir()
