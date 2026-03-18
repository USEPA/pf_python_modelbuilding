import re
from typing import Any, Callable


_INCHI_KEY_PATTERN = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")


def normalize_inchi_key(value: Any) -> str | None:
    if not isinstance(value, str):
        return None

    inchi_key = value.strip().upper()
    if not inchi_key or inchi_key == "N/A":
        return None

    if not _INCHI_KEY_PATTERN.fullmatch(inchi_key):
        return None

    return inchi_key


def ensure_chemical_inchi_key(
    chemical: Any,
    smiles_to_inchi_key: Callable[[Any], str | None],
    fallback_smiles: str | None = None,
):
    if not isinstance(chemical, dict):
        return chemical

    chemical_with_inchi = dict(chemical)
    inchi_key = normalize_inchi_key(chemical_with_inchi.get("inchiKey"))

    if inchi_key is None:
        for candidate_smiles in (
            chemical_with_inchi.get("canonicalSmiles"),
            chemical_with_inchi.get("smiles"),
            fallback_smiles,
        ):
            inchi_key = normalize_inchi_key(smiles_to_inchi_key(candidate_smiles))
            if inchi_key is not None:
                break

    chemical_with_inchi["inchiKey"] = inchi_key
    return chemical_with_inchi


def build_prediction_cache_key(
    model_id: Any,
    smiles_to_inchi_key: Callable[[Any], str | None],
    *,
    smiles: Any = None,
    chemical: Any = None,
) -> str | None:
    if isinstance(chemical, dict):
        chemical_with_inchi = ensure_chemical_inchi_key(
            chemical,
            smiles_to_inchi_key,
            fallback_smiles=None if smiles is None else str(smiles).strip(),
        )
        inchi_key = normalize_inchi_key(chemical_with_inchi.get("inchiKey"))
    else:
        inchi_key = normalize_inchi_key(smiles_to_inchi_key(smiles))

    if inchi_key is None:
        return None

    return f"{inchi_key}-{model_id}"
