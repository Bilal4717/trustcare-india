import json
import os
from typing import Any, List, Optional

import pandas as pd
from langchain_core.documents import Document

from healthcare_app.config import (
    CHUNK_BATCH_SIZE,
    DATASET_PATH,
    DATA_SOURCE,
    DATABRICKS_EXPORT_PATH,
    VECTORSTORE_PATH as DEFAULT_VECTORSTORE_PATH,
)

_NULL_SENTINELS = {"null", "none", "[]", "{}", "", "nan", "nat"}


def _is_empty(val) -> bool:
    if val is None:
        return True
    return str(val).strip().lower() in _NULL_SENTINELS


def _parse_json_list(val) -> List[str]:
    if _is_empty(val):
        return []
    try:
        parsed = json.loads(str(val))
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if x]
    except Exception:
        pass
    raw = str(val).strip().strip("[]").strip()
    if not raw:
        return []
    return [x.strip().strip('"').strip("'") for x in raw.split(",") if x.strip()]


def _read_any_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl", dtype=str)
    if ext == ".csv":
        return pd.read_csv(path, dtype=str)
    if ext == ".json":
        return pd.read_json(path, dtype=str)
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset extension: {ext}")


def _normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Databricks-exported schemas into the app's canonical columns.
    This lets upstream ETL evolve while keeping agent logic stable.
    """
    aliases = {
        "facility_name": "name",
        "facility_type": "facilityTypeId",
        "city": "address_city",
        "state": "address_stateOrRegion",
        "postcode": "address_zipOrPostcode",
        "zip": "address_zipOrPostcode",
        "pincode": "address_zipOrPostcode",
        "procedures": "procedure",
        "doctors_count": "numberDoctors",
        "beds": "capacity",
        "followers_count": "engagement_metrics_n_followers",
        "staff_affiliated": "affiliated_staff_presence",
        "has_custom_logo": "custom_logo_presence",
    }
    rename_map = {src: dst for src, dst in aliases.items() if src in df.columns and dst not in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = [
        "name",
        "facilityTypeId",
        "address_city",
        "address_stateOrRegion",
        "address_zipOrPostcode",
        "description",
        "specialties",
        "capability",
        "procedure",
        "equipment",
        "numberDoctors",
        "capacity",
        "latitude",
        "longitude",
        "engagement_metrics_n_followers",
        "affiliated_staff_presence",
        "custom_logo_presence",
    ]
    for c in expected:
        if c not in df.columns:
            df[c] = None
    return df


def load_dataset() -> pd.DataFrame:
    source = DATA_SOURCE
    if source == "databricks_export":
        raw_path = DATABRICKS_EXPORT_PATH
    else:
        raw_path = DATASET_PATH

    df = _read_any_table(raw_path)
    df = _normalize_schema(df)
    df = _ensure_required_columns(df)
    sentinel_values = ["null", "None", "NaN", "[]", "{}", "nan", "NaT"]
    for col in df.columns:
        df[col] = df[col].apply(lambda x: None if (x is None or str(x).strip() in sentinel_values) else x)

    for col in ["latitude", "longitude", "engagement_metrics_n_followers"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    import re as _re

    def _clean_pin(x):
        if x is None or str(x).strip() in ("nan", "None", ""):
            return None
        digits = _re.sub(r"[^0-9]", "", str(x))
        return digits.zfill(6)[:6] if len(digits) >= 5 else None

    df["pin"] = df["address_zipOrPostcode"].apply(_clean_pin)

    key_fields = ["description", "specialties", "capability", "procedure", "equipment", "numberDoctors"]
    df["_completeness"] = df[key_fields].apply(
        lambda row: sum(1 for v in row if not _is_empty(v)) / len(key_fields),
        axis=1,
    )
    return df


def build_facility_text(row: pd.Series) -> str:
    parts = []
    if not _is_empty(row.get("name")):
        parts.append(f"Facility name: {row['name']}")
    if not _is_empty(row.get("facilityTypeId")):
        parts.append(f"Facility type: {row.get('facilityTypeId')}")
    city = row.get("address_city") or ""
    state = row.get("address_stateOrRegion") or ""
    if not _is_empty(city) or not _is_empty(state):
        parts.append(f"Location: {city}, {state}".strip(", "))
    if not _is_empty(row.get("pin")):
        parts.append(f"PIN code: {row.get('pin')}")
    if not _is_empty(row.get("description")):
        parts.append(f"Description: {row.get('description')}")

    for field_name, label in [
        ("specialties", "Specialties"),
        ("procedure", "Procedures"),
        ("equipment", "Equipment"),
        ("capability", "Capabilities"),
    ]:
        values = _parse_json_list(row.get(field_name))
        if values:
            parts.append(f"{label}: {', '.join(values)}")

    if not _is_empty(row.get("numberDoctors")):
        parts.append(f"Number of doctors: {row.get('numberDoctors')}")
    if not _is_empty(row.get("capacity")):
        parts.append(f"Bed capacity: {row.get('capacity')}")
    return " | ".join(parts)


def _build_metadata(idx: int, row: pd.Series) -> dict:
    lat = row.get("latitude")
    lon = row.get("longitude")
    return {
        "facility_id": idx,
        "name": str(row.get("name") or ""),
        "type": str(row.get("facilityTypeId") or ""),
        "city": str(row.get("address_city") or ""),
        "state": str(row.get("address_stateOrRegion") or ""),
        "pin": str(row.get("pin") or ""),
        "lat": float(lat) if lat is not None and not pd.isna(lat) else None,
        "lon": float(lon) if lon is not None and not pd.isna(lon) else None,
        "description": str(row.get("description") or ""),
        "specialties": str(row.get("specialties") or ""),
        "procedures": str(row.get("procedure") or ""),
        "equipment": str(row.get("equipment") or ""),
        "capability": str(row.get("capability") or ""),
        "num_doctors": str(row.get("numberDoctors") or ""),
        "capacity": str(row.get("capacity") or ""),
        "completeness": float(row.get("_completeness") or 0.0),
        "affiliated_staff": str(row.get("affiliated_staff_presence")).lower() == "true",
        "custom_logo": str(row.get("custom_logo_presence")).lower() == "true",
        "followers": float(row.get("engagement_metrics_n_followers") or 0),
    }


def get_vectorstore(df: pd.DataFrame, embedding, vectorstore_path: Optional[str] = None) -> Any:
    try:
        from langchain_community.vectorstores import FAISS
    except Exception as exc:
        raise RuntimeError(
            "FAISS backend unavailable in this runtime. "
            "Use a cloud retriever or keyword fallback mode."
        ) from exc

    persist = vectorstore_path or DEFAULT_VECTORSTORE_PATH
    if os.path.exists(persist):
        return FAISS.load_local(persist, embedding, allow_dangerous_deserialization=True)

    docs = []
    for idx, row in df.iterrows():
        text = build_facility_text(row)
        if len(text.strip()) < 20:
            continue
        docs.append(Document(page_content=text, metadata=_build_metadata(idx, row)))

    vectorstore = None
    for i in range(0, len(docs), CHUNK_BATCH_SIZE):
        batch = docs[i : i + CHUNK_BATCH_SIZE]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embedding)
        else:
            vectorstore.add_documents(batch)

    if vectorstore is None:
        raise RuntimeError("Vector store build failed")
    vectorstore.save_local(persist)
    return vectorstore


def format_docs_for_llm(docs: list) -> str:
    if not docs:
        return "No matching facilities found."
    lines = []
    for i, doc in enumerate(docs, start=1):
        m = doc.metadata
        lines.append(
            f"[{i}] {m.get('name', 'Unknown')} ({m.get('type', '')})\n"
            f"    Location: {m.get('city', '')}, {m.get('state', '')} — PIN {m.get('pin', '')}\n"
            f"    {doc.page_content[:400]}..."
        )
    return "\n\n".join(lines)

