import os
from urllib.parse import urlencode


DEFAULT_CIM_RENDER_URL = "https://cim-dev.sciencedataexperts.com/api/resolver/render"
DEFAULT_CIM_RENDER_FORMAT = "PNG"


def get_render_service_url() -> str:
    value = os.getenv("CIM_RENDER_URL", DEFAULT_CIM_RENDER_URL).strip()
    return value or DEFAULT_CIM_RENDER_URL


def get_render_smiles(chemical: dict | None) -> str | None:
    if not isinstance(chemical, dict):
        return None

    for field_name in ("smiles", "canonicalSmiles"):
        value = chemical.get(field_name)
        if isinstance(value, str):
            normalized = value.strip()
            if normalized:
                return normalized

    return None


def build_render_image_url(
    smiles: str | None,
    *,
    width: int | None = None,
    height: int | None = None,
    image_format: str = DEFAULT_CIM_RENDER_FORMAT,
) -> str | None:
    if not isinstance(smiles, str):
        return None

    normalized_smiles = smiles.strip()
    if not normalized_smiles:
        return None

    params = {"smiles": normalized_smiles, "format": image_format.upper()}

    if width is not None:
        params["width"] = width

    if height is not None:
        params["height"] = height

    base_url = get_render_service_url()
    separator = "&" if "?" in base_url else "?"
    return f"{base_url}{separator}{urlencode(params)}"


def resolve_report_image_src(
    chemical: dict | None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> str | None:
    if not isinstance(chemical, dict):
        return None

    existing_image_src = chemical.get("imageSrc")
    if existing_image_src is not None:
        return existing_image_src

    return build_render_image_url(
        get_render_smiles(chemical),
        width=width,
        height=height,
    )
