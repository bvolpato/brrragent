import base64

import pytest

from brrragent import ImageInput
from brrragent.images import (
    _chat_user_content,
    _decode_data_url,
    _responses_user_content,
)


def test_image_input_builds_data_urls_from_bytes_and_files(tmp_path):
    expected = base64.b64encode(b"png-data").decode("ascii")
    direct = ImageInput.from_bytes(
        b"png-data", media_type="image/png", detail="original"
    )
    path = tmp_path / "sample.png"
    path.write_bytes(b"png-data")
    from_file = ImageInput.from_file(path, detail="high")

    assert direct.url == f"data:image/png;base64,{expected}"
    assert direct.detail == "original"
    assert from_file.url == f"data:image/png;base64,{expected}"
    assert from_file.media_type == "image/png"
    assert _decode_data_url(direct.url) == ("image/png", b"png-data")


def test_image_input_validates_source_and_detail():
    with pytest.raises(ValueError, match="must not be empty"):
        ImageInput("")
    with pytest.raises(ValueError, match="must start with image/"):
        ImageInput.from_base64("aW1hZ2U=", media_type="text/plain")
    with pytest.raises(ValueError, match="invalid Base64"):
        ImageInput.from_base64("not-base64!", media_type="image/png")
    with pytest.raises(ValueError, match="invalid Base64"):
        _decode_data_url("data:image/png;base64,not-base64!")
    with pytest.raises(ValueError, match="Image detail"):
        ImageInput("https://example.test/image.png", detail="tiny")


def test_image_content_builders_preserve_text_only_shape_and_multiple_images():
    images = (
        ImageInput("https://example.test/one.png", detail="low"),
        ImageInput("https://example.test/two.jpg", detail="high"),
    )

    assert _responses_user_content("describe", ()) == "describe"
    assert _chat_user_content("describe", ()) == "describe"
    assert _responses_user_content("describe", images) == [
        {"type": "input_text", "text": "describe"},
        {
            "type": "input_image",
            "image_url": "https://example.test/one.png",
            "detail": "low",
        },
        {
            "type": "input_image",
            "image_url": "https://example.test/two.jpg",
            "detail": "high",
        },
    ]
    assert _chat_user_content("describe", images)[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/one.png", "detail": "low"},
    }
