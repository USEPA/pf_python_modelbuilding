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


def _chemical_inchi_key(
    chemical: Any,
    smiles_to_inchi_key: Callable[[Any], str | None],
) -> str | None:
    if not isinstance(chemical, dict):
        return None

    inchi_key = normalize_inchi_key(chemical.get("inchiKey"))
    if inchi_key is not None:
        return inchi_key

    for candidate_smiles in (
        chemical.get("canonicalSmiles"),
        chemical.get("smiles"),
    ):
        inchi_key = normalize_inchi_key(smiles_to_inchi_key(candidate_smiles))
        if inchi_key is not None:
            return inchi_key

    return None


def standardized_chemical_changes_identity(
    input_smiles: Any,
    standardized_chemical: Any,
    smiles_to_inchi_key: Callable[[Any], str | None],
) -> bool:
    if not isinstance(standardized_chemical, dict):
        return False

    input_inchi_key = normalize_inchi_key(smiles_to_inchi_key(input_smiles))
    standardized_inchi_key = _chemical_inchi_key(standardized_chemical, smiles_to_inchi_key)

    if input_inchi_key and standardized_inchi_key:
        return input_inchi_key != standardized_inchi_key

    if input_inchi_key or standardized_inchi_key:
        return True

    original_smiles = str(input_smiles).strip() if isinstance(input_smiles, str) else ""
    standardized_smiles = (
        standardized_chemical.get("canonicalSmiles") or standardized_chemical.get("smiles")
    )
    standardized_smiles = (
        standardized_smiles.strip() if isinstance(standardized_smiles, str) else ""
    )

    return bool(original_smiles and standardized_smiles and standardized_smiles != original_smiles)


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
