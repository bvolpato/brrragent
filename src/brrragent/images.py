"""Provider-neutral image inputs for multimodal requests."""

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ImageDetail = Literal["auto", "low", "high", "original"]
_IMAGE_DETAILS = {"auto", "low", "high", "original"}


@dataclass(frozen=True, slots=True)
class ImageInput:
    """Image URL or Base64 data URL sent with a model request."""

    url: str
    detail: ImageDetail = "auto"
    media_type: str | None = None

    def __post_init__(self) -> None:
        url = self.url.strip()
        if not url:
            raise ValueError("Image URL must not be empty")
        object.__setattr__(self, "url", url)
        if self.detail not in _IMAGE_DETAILS:
            raise ValueError("Image detail must be one of: auto, low, high, original")

    @classmethod
    def from_base64(
        cls,
        data: str,
        *,
        media_type: str,
        detail: ImageDetail = "auto",
    ) -> "ImageInput":
        encoded = data.strip()
        if not encoded:
            raise ValueError("Base64 image data must not be empty")
        if not media_type.startswith("image/"):
            raise ValueError("Image media type must start with image/")
        try:
            base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise ValueError("Image data contains invalid Base64") from exc
        return cls(
            url=f"data:{media_type};base64,{encoded}",
            detail=detail,
            media_type=media_type,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        media_type: str,
        detail: ImageDetail = "auto",
    ) -> "ImageInput":
        if not data:
            raise ValueError("Image bytes must not be empty")
        encoded = base64.b64encode(data).decode("ascii")
        return cls.from_base64(encoded, media_type=media_type, detail=detail)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        media_type: str | None = None,
        detail: ImageDetail = "auto",
    ) -> "ImageInput":
        image_path = Path(path)
        resolved_media_type = media_type or mimetypes.guess_type(image_path.name)[0]
        if not resolved_media_type:
            raise ValueError("Could not infer image media type")
        return cls.from_bytes(
            image_path.read_bytes(),
            media_type=resolved_media_type,
            detail=detail,
        )


def _responses_user_content(
    user_prompt: str, images: tuple[ImageInput, ...]
) -> str | list[dict]:
    if not images:
        return user_prompt
    return [
        {"type": "input_text", "text": user_prompt},
        *[
            {
                "type": "input_image",
                "image_url": image.url,
                "detail": image.detail,
            }
            for image in images
        ],
    ]


def _chat_user_content(
    user_prompt: str, images: tuple[ImageInput, ...]
) -> str | list[dict]:
    if not images:
        return user_prompt
    return [
        {"type": "text", "text": user_prompt},
        *[
            {
                "type": "image_url",
                "image_url": {"url": image.url, "detail": image.detail},
            }
            for image in images
        ],
    ]


def _decode_data_url(url: str) -> tuple[str, bytes] | None:
    if not url.startswith("data:"):
        return None
    header, separator, encoded = url.partition(",")
    if not separator or not header.endswith(";base64"):
        raise ValueError("Image data URL must contain Base64 data")
    media_type = header[5:-7]
    if not media_type.startswith("image/"):
        raise ValueError("Image data URL media type must start with image/")
    try:
        return media_type, base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise ValueError("Image data URL contains invalid Base64") from exc
