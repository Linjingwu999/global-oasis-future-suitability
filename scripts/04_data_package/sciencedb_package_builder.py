from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import struct
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REGIONS: tuple[tuple[str, str], ...] = (
    ("\u963f\u62c9\u4f2f\u534a\u5c9b", "arabian_peninsula"),
    ("\u5317\u7f8e\u6d32", "north_america"),
    ("\u5927\u6d0b\u6d32", "oceania"),
    ("\u975e\u6d32\u5317\u90e8", "north_africa"),
    ("\u975e\u6d32\u5357\u90e8", "south_africa"),
    ("\u5357\u7f8e\u6d32", "south_america"),
    ("\u4e9a\u6d32\u4e1c\u90e8", "east_asia"),
    ("\u4e9a\u6d32\u897f\u5357\u90e8", "southwest_asia"),
    ("\u4e9a\u6d32\u4e2d\u90e8", "central_asia"),
)

REQUIRED_SUFFIXES = (".shp", ".shx", ".dbf", ".prj", ".cpg")
OPTIONAL_SUFFIXES = (".shp.xml", ".sbn", ".sbx")
EXPECTED_TOTAL_FEATURES = 3443
DATASET_DIRNAME = "global_oasis_regions_final_20260527"
ARCHIVE_NAME = "global_oasis_regions_final_20260527_sdb.zip"


@dataclass
class RegionValidation:
    source_name: str
    archive_name: str
    source_dir: str
    status: str = "pending"
    feature_count: int | None = None
    dbf_count: int | None = None
    shape_type: int | None = None
    files: list[dict[str, Any]] = field(default_factory=list)
    missing_required: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    excluded_zip_files: list[str] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_exact(path: Path, size: int) -> bytes:
    with path.open("rb") as file_obj:
        data = file_obj.read(size)
    if len(data) != size:
        raise ValueError(f"{path.name} is shorter than {size} bytes")
    return data


def read_be_i32(data: bytes, offset: int) -> int:
    return struct.unpack(">i", data[offset : offset + 4])[0]


def read_le_i32(data: bytes, offset: int) -> int:
    return struct.unpack("<i", data[offset : offset + 4])[0]


def read_le_u16(data: bytes, offset: int) -> int:
    return struct.unpack("<H", data[offset : offset + 2])[0]


def read_le_u32(data: bytes, offset: int) -> int:
    return struct.unpack("<I", data[offset : offset + 4])[0]


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def is_ascii_path(path: str) -> bool:
    try:
        path.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


def ensure_clean_archive_name(name: str) -> None:
    if not name.lower().endswith(".zip"):
        raise ValueError("Archive name must end with .zip")
    if not is_ascii_path(name):
        raise ValueError(f"Archive name is not ASCII-only: {name}")
    if any(part in name for part in ("\\", "/", ":")):
        raise ValueError(f"Archive name must be a plain file name, got: {name}")


def setup_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "package_sdb_oasis_regions.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_path


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)
            file_obj.write("\n")
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as file_obj:
            file_obj.write(text)
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def update_status(status_path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(status_path, payload)


def parse_shp_header(path: Path) -> dict[str, Any]:
    data = read_exact(path, 100)
    size = path.stat().st_size
    return {
        "file_code": read_be_i32(data, 0),
        "header_file_length_bytes": read_be_i32(data, 24) * 2,
        "version": read_le_i32(data, 28),
        "shape_type": read_le_i32(data, 32),
        "actual_size_bytes": size,
        "size_matches_header": size == read_be_i32(data, 24) * 2,
    }


def parse_dbf_header(path: Path) -> dict[str, Any]:
    data = read_exact(path, 32)
    size = path.stat().st_size
    row_count = read_le_u32(data, 4)
    header_length = read_le_u16(data, 8)
    row_length = read_le_u16(data, 10)
    expected_min_size = header_length + row_count * row_length
    return {
        "row_count": row_count,
        "header_length": header_length,
        "row_length": row_length,
        "actual_size_bytes": size,
        "expected_min_size_bytes": expected_min_size,
        "size_at_least_expected": size >= expected_min_size,
    }


def parse_shx_records(shx_path: Path, shp_size: int) -> dict[str, Any]:
    header = parse_shp_header(shx_path)
    shx_size = shx_path.stat().st_size
    if shx_size < 100:
        raise ValueError(f"{shx_path.name} is shorter than a Shapefile header")
    if (shx_size - 100) % 8 != 0:
        raise ValueError(f"{shx_path.name} length is not compatible with .shx records")

    offsets_ok = True
    monotonic_ok = True
    bounds_ok = True
    previous_offset = 0
    last_record_end = 100
    count = 0

    with shx_path.open("rb") as file_obj:
        file_obj.seek(100)
        while True:
            record = file_obj.read(8)
            if not record:
                break
            if len(record) != 8:
                offsets_ok = False
                break
            offset_bytes = struct.unpack(">i", record[:4])[0] * 2
            content_bytes = struct.unpack(">i", record[4:])[0] * 2
            record_end = offset_bytes + 8 + content_bytes
            if offset_bytes < 100 or content_bytes < 4:
                offsets_ok = False
            if offset_bytes < previous_offset:
                monotonic_ok = False
            if record_end > shp_size:
                bounds_ok = False
            previous_offset = offset_bytes
            last_record_end = max(last_record_end, record_end)
            count += 1

    return {
        **header,
        "record_count": count,
        "offsets_ok": offsets_ok,
        "monotonic_offsets": monotonic_ok,
        "records_within_shp_size": bounds_ok,
        "last_record_end_bytes": last_record_end,
    }


def source_file_for_suffix(region_dir: Path, source_name: str, suffix: str) -> Path:
    if suffix == ".shp.xml":
        return region_dir / f"{source_name}.shp.xml"
    return region_dir / f"{source_name}{suffix}"


def archive_name_for_suffix(region_en: str, suffix: str) -> str:
    if suffix == ".shp.xml":
        return f"{region_en}.shp.xml"
    return f"{region_en}{suffix}"


def validate_region(data_root: Path, source_name: str, archive_name: str) -> RegionValidation:
    region_dir = data_root / source_name
    result = RegionValidation(
        source_name=source_name,
        archive_name=archive_name,
        source_dir=str(region_dir),
    )
    logging.info("Validating region %s -> %s", source_name, archive_name)
    try:
        if not region_dir.exists():
            result.errors.append(f"Missing source directory: {region_dir}")
            result.status = "failed"
            return result
        if not region_dir.is_dir():
            result.errors.append(f"Source path is not a directory: {region_dir}")
            result.status = "failed"
            return result

        for zip_path in sorted(region_dir.glob("*.zip")):
            result.excluded_zip_files.append(zip_path.name)

        for suffix in REQUIRED_SUFFIXES:
            path = source_file_for_suffix(region_dir, source_name, suffix)
            if not path.exists():
                result.missing_required.append(path.name)

        if result.missing_required:
            result.errors.append("Missing required Shapefile components")
            result.status = "failed"
            return result

        shp_path = source_file_for_suffix(region_dir, source_name, ".shp")
        shx_path = source_file_for_suffix(region_dir, source_name, ".shx")
        dbf_path = source_file_for_suffix(region_dir, source_name, ".dbf")

        shp = parse_shp_header(shp_path)
        shx = parse_shx_records(shx_path, shp_path.stat().st_size)
        dbf = parse_dbf_header(dbf_path)

        if shp["file_code"] != 9994 or shx["file_code"] != 9994:
            result.errors.append("Invalid Shapefile header file code")
        if shp["version"] != 1000 or shx["version"] != 1000:
            result.errors.append("Unexpected Shapefile version")
        if shp["shape_type"] != shx["shape_type"]:
            result.errors.append("Shape type mismatch between .shp and .shx")
        if not shp["size_matches_header"]:
            result.errors.append(".shp file size does not match header length")
        if not shx["size_matches_header"]:
            result.errors.append(".shx file size does not match header length")
        if not shx["offsets_ok"]:
            result.errors.append(".shx record offsets contain invalid values")
        if not shx["monotonic_offsets"]:
            result.errors.append(".shx record offsets are not monotonic")
        if not shx["records_within_shp_size"]:
            result.errors.append(".shx records point outside .shp file")
        if not dbf["size_at_least_expected"]:
            result.errors.append(".dbf file size is smaller than row count implies")
        if shx["record_count"] != dbf["row_count"]:
            result.errors.append(".shx feature count does not match .dbf row count")

        result.feature_count = int(shx["record_count"])
        result.dbf_count = int(dbf["row_count"])
        result.shape_type = int(shp["shape_type"])

        for suffix in REQUIRED_SUFFIXES + OPTIONAL_SUFFIXES:
            path = source_file_for_suffix(region_dir, source_name, suffix)
            if not path.exists():
                if suffix in OPTIONAL_SUFFIXES:
                    result.warnings.append(f"Optional component absent: {path.name}")
                continue
            archive_rel = f"{DATASET_DIRNAME}/{archive_name}/{archive_name_for_suffix(archive_name, suffix)}"
            result.files.append(
                {
                    "source_path": str(path),
                    "archive_path": archive_rel,
                    "size_bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                    "ascii_archive_path": is_ascii_path(archive_rel),
                }
            )

        if any(not item["ascii_archive_path"] for item in result.files):
            result.errors.append("At least one archive path is not ASCII-only")

        result.status = "failed" if result.errors else "success"
    except Exception as exc:  # noqa: BLE001
        logging.exception("Validation failed for %s", source_name)
        result.status = "failed"
        result.errors.append(f"{type(exc).__name__}: {exc}")
    return result


def write_manifest(path: Path, validations: list[RegionValidation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    fieldnames = [
        "source_region_name",
        "archive_region_name",
        "feature_count",
        "source_path",
        "archive_path",
        "size_bytes",
        "sha256",
    ]
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            for validation in validations:
                for item in validation.files:
                    writer.writerow(
                        {
                            "source_region_name": validation.source_name,
                            "archive_region_name": validation.archive_name,
                            "feature_count": validation.feature_count,
                            "source_path": item["source_path"],
                            "archive_path": item["archive_path"],
                            "size_bytes": item["size_bytes"],
                            "sha256": item["sha256"],
                        }
                    )
        Path(temp_name).replace(path)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise


def validation_summary_text(report: dict[str, Any]) -> str:
    lines = [
        "Science Data Bank upload package validation summary",
        f"Generated at: {report['generated_at']}",
        f"Source root: {report['source_root']}",
        f"Archive path: {report['archive_path']}",
        f"Expected feature count: {report['expected_total_features']}",
        f"Observed feature count: {report['observed_total_features']}",
        f"Feature count matches expected: {report['feature_count_matches_expected']}",
        f"Archive entry names ASCII-only: {report.get('archive_entry_names_ascii_only')}",
        "",
        "Per-region feature counts:",
    ]
    for region in report["regions"]:
        lines.append(
            f"- {region['archive_name']}: {region.get('feature_count')} "
            f"(status={region['status']}, dbf_count={region.get('dbf_count')})"
        )
    lines.extend(["", "Warnings and errors:"])
    found = False
    for region in report["regions"]:
        for warning in region.get("warnings", []):
            found = True
            lines.append(f"- WARNING {region['archive_name']}: {warning}")
        for error in region.get("errors", []):
            found = True
            lines.append(f"- ERROR {region['archive_name']}: {error}")
    if not found:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def serialize_validation(validation: RegionValidation) -> dict[str, Any]:
    return {
        "source_name": validation.source_name,
        "archive_name": validation.archive_name,
        "source_dir": validation.source_dir,
        "status": validation.status,
        "feature_count": validation.feature_count,
        "dbf_count": validation.dbf_count,
        "shape_type": validation.shape_type,
        "files": validation.files,
        "missing_required": validation.missing_required,
        "warnings": validation.warnings,
        "errors": validation.errors,
        "excluded_zip_files": validation.excluded_zip_files,
    }


def add_file_to_zip(zip_obj: zipfile.ZipFile, source_path: Path, archive_path: str) -> None:
    if not is_ascii_path(archive_path):
        raise ValueError(f"Archive path is not ASCII-only: {archive_path}")
    zip_obj.write(source_path, archive_path)


def write_archive(
    archive_path: Path,
    validations: list[RegionValidation],
    metadata_paths: list[Path],
    force: bool,
    status_path: Path,
    status_payload: dict[str, Any],
) -> dict[str, Any]:
    if archive_path.exists() and not force:
        logging.info("Archive already exists and --force is not set: %s", archive_path)
        with zipfile.ZipFile(archive_path, "r") as zip_obj:
            bad_file = zip_obj.testzip()
            names = zip_obj.namelist()
        return {
            "status": "success",
            "skipped_existing": True,
            "testzip_bad_file": bad_file,
            "entry_count": len(names),
            "archive_size_bytes": archive_path.stat().st_size,
            "sha256": file_sha256(archive_path),
            "archive_entry_names_ascii_only": all(is_ascii_path(name) for name in names),
        }

    temp_archive = archive_path.with_name(archive_path.name + ".tmp")
    temp_archive.unlink(missing_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    status_payload["package"] = {
        "status": "running",
        "archive_path": str(archive_path),
        "started_at": now_iso(),
        "updated_at": now_iso(),
    }
    update_status(status_path, status_payload)

    try:
        with zipfile.ZipFile(
            temp_archive,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=6,
            allowZip64=True,
        ) as zip_obj:
            for validation in validations:
                for item in validation.files:
                    source_path = Path(item["source_path"])
                    archive_rel = item["archive_path"]
                    logging.info("Adding %s", archive_rel)
                    add_file_to_zip(zip_obj, source_path, archive_rel)
            for metadata_path in metadata_paths:
                archive_rel = f"{DATASET_DIRNAME}/metadata/{metadata_path.name}"
                logging.info("Adding %s", archive_rel)
                add_file_to_zip(zip_obj, metadata_path, archive_rel)

        with zipfile.ZipFile(temp_archive, "r") as zip_obj:
            bad_file = zip_obj.testzip()
            names = zip_obj.namelist()
        if bad_file is not None:
            raise ValueError(f"Zip integrity test failed at entry: {bad_file}")
        if not all(is_ascii_path(name) for name in names):
            raise ValueError("Archive contains non-ASCII entry names")

        temp_archive.replace(archive_path)
        elapsed = round(time.time() - start_time, 1)
        return {
            "status": "success",
            "skipped_existing": False,
            "testzip_bad_file": None,
            "entry_count": len(names),
            "archive_size_bytes": archive_path.stat().st_size,
            "sha256": file_sha256(archive_path),
            "archive_entry_names_ascii_only": True,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        logging.exception("Archive creation failed")
        temp_archive.unlink(missing_ok=True)
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "archive_entry_names_ascii_only": None,
        }


def parse_args() -> argparse.Namespace:
    default_data_root = Path.home() / "Desktop" / "\u7eff\u6d32\u7f16\u7801\u6700\u7ec8\u7248"
    parser = argparse.ArgumentParser(
        description="Validate and package the nine final global oasis Shapefile regions for Science Data Bank upload."
    )
    parser.add_argument("--data-root", type=Path, default=default_data_root)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--archive-name", default=ARCHIVE_NAME)
    parser.add_argument("--expected-count", type=int, default=EXPECTED_TOTAL_FEATURES)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing final archive.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (args.output_dir or (data_root / "sdb_upload_package_20260601")).expanduser().resolve()
    ensure_clean_archive_name(args.archive_name)
    archive_path = output_dir / args.archive_name
    log_path = setup_logging(output_dir)
    status_path = output_dir / "processing_status.json"
    manifest_path = output_dir / "manifest.csv"
    report_path = output_dir / "validation_report.json"
    summary_path = output_dir / "validation_summary.txt"
    contents_path = output_dir / "archive_contents.txt"

    status_payload: dict[str, Any] = {
        "task": "package_sdb_oasis_regions",
        "status": "running",
        "started_at": now_iso(),
        "updated_at": now_iso(),
        "source_root": str(data_root),
        "output_dir": str(output_dir),
        "archive_path": str(archive_path),
        "regions": [],
        "package": {"status": "pending"},
    }
    update_status(status_path, status_payload)

    logging.info("Source root: %s", data_root)
    logging.info("Output directory: %s", output_dir)
    logging.info("Archive path: %s", archive_path)

    validations: list[RegionValidation] = []
    for source_name, archive_name in REGIONS:
        region_status = {
            "source_name": source_name,
            "archive_name": archive_name,
            "status": "running",
            "started_at": now_iso(),
        }
        status_payload["regions"].append(region_status)
        status_payload["updated_at"] = now_iso()
        update_status(status_path, status_payload)

        validation = validate_region(data_root, source_name, archive_name)
        validations.append(validation)
        region_status.update(
            {
                "status": validation.status,
                "ended_at": now_iso(),
                "feature_count": validation.feature_count,
                "dbf_count": validation.dbf_count,
                "errors": validation.errors,
                "warnings": validation.warnings,
            }
        )
        status_payload["updated_at"] = now_iso()
        update_status(status_path, status_payload)

    observed_total = sum(v.feature_count or 0 for v in validations)
    all_regions_success = all(v.status == "success" for v in validations)
    report: dict[str, Any] = {
        "generated_at": now_iso(),
        "source_root": str(data_root),
        "output_dir": str(output_dir),
        "archive_path": str(archive_path),
        "log_path": str(log_path),
        "status_path": str(status_path),
        "expected_total_features": args.expected_count,
        "observed_total_features": observed_total,
        "feature_count_matches_expected": observed_total == args.expected_count,
        "all_regions_success": all_regions_success,
        "regions": [serialize_validation(v) for v in validations],
    }
    planned_archive_paths = [
        item["archive_path"]
        for validation in validations
        for item in validation.files
    ]
    planned_archive_paths.extend(
        f"{DATASET_DIRNAME}/metadata/{path.name}"
        for path in (manifest_path, report_path, summary_path)
    )
    report["planned_archive_entry_count"] = len(planned_archive_paths)
    report["archive_entry_names_ascii_only"] = all(
        is_ascii_path(path) for path in planned_archive_paths
    )

    write_manifest(manifest_path, validations)
    write_json_atomic(report_path, report)
    write_text_atomic(summary_path, validation_summary_text(report))

    if not all_regions_success or observed_total != args.expected_count:
        status_payload["status"] = "failed"
        status_payload["ended_at"] = now_iso()
        status_payload["updated_at"] = now_iso()
        status_payload["package"] = {
            "status": "failed",
            "reason": "Validation failed or feature count mismatch; archive was not created.",
        }
        update_status(status_path, status_payload)
        logging.error("Validation failed; archive was not created.")
        return 1

    package_result = write_archive(
        archive_path=archive_path,
        validations=validations,
        metadata_paths=[manifest_path, report_path, summary_path],
        force=args.force,
        status_path=status_path,
        status_payload=status_payload,
    )
    report["package"] = package_result
    report["archive_entry_names_ascii_only"] = package_result.get("archive_entry_names_ascii_only")
    write_json_atomic(report_path, report)
    write_text_atomic(summary_path, validation_summary_text(report))

    if package_result["status"] == "success":
        with zipfile.ZipFile(archive_path, "r") as zip_obj:
            write_text_atomic(contents_path, "\n".join(zip_obj.namelist()) + "\n")
        status_payload["status"] = "success"
    else:
        status_payload["status"] = "failed"
    status_payload["ended_at"] = now_iso()
    status_payload["updated_at"] = now_iso()
    status_payload["package"] = package_result
    update_status(status_path, status_payload)

    if package_result["status"] != "success":
        logging.error("Packaging failed: %s", package_result.get("error"))
        return 1

    logging.info("Done. Archive: %s", archive_path)
    logging.info("Observed feature count: %s", observed_total)
    logging.info("Archive SHA256: %s", package_result["sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
