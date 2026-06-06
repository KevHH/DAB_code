#!/usr/bin/env python
"""Convert QM-sym C4h XYZ archive members to pickled ASE Atoms objects."""

from __future__ import annotations

import argparse
import json
import pickle
import tarfile
from pathlib import Path, PurePosixPath

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from tqdm import tqdm


ARCHIVE_CONFIGS = {
    "QM_sym_C4h_1.tar": {
        "default_out": Path("data/qm_sym_c4h_1/qm_sym_c4h_1_ase_u0_ha.pkl"),
        "expected_count": 12217,
    },
    "QM_sym_C4h_2.tar": {
        "default_out": Path("data/qm_sym_c4h_2/qm_sym_c4h_2_ase_u0_ha.pkl"),
        "expected_count": 13451,
    },
}
DEFAULT_ARCHIVE_NAME = "QM_sym_C4h_1.tar"
DEFAULT_ARCHIVE = Path("QM-sym-database") / DEFAULT_ARCHIVE_NAME
TARGET_FIELD_INDEX = 15
TARGET_NAME = "sum_electronic_zero_point_energy_ha"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        default=str(DEFAULT_ARCHIVE),
        help="Input QM-sym C4h archive.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output ASE pickle path. Default is based on the archive name.",
    )
    parser.add_argument("--cell-size", type=float, default=30.0, help="Cubic cell size in Angstrom.")
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Expected number of XYZ members. Default is archive-specific; use 0 to skip the check.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    return parser.parse_args()


def parse_float(value: str, field: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Could not parse {field}={value!r} as float.") from exc


def parse_properties(line: str) -> dict:
    fields = [field.strip() for field in line.strip().split("|")]
    if len(fields) != 45:
        raise ValueError(f"Expected 45 pipe-delimited property fields, got {len(fields)}.")

    return {
        "symmetry_group": fields[0],
        "band_gap_ha": parse_float(fields[1], "band_gap_ha"),
        "lumo_ha": parse_float(fields[2], "lumo_ha"),
        "homo_ha": parse_float(fields[3], "homo_ha"),
        "rotational_constants_ghz": [parse_float(value, "rotational_constants_ghz") for value in fields[4:7]],
        "dipole_moment_d": {
            "x": parse_float(fields[7], "dipole_moment_d.x"),
            "y": parse_float(fields[8], "dipole_moment_d.y"),
            "z": parse_float(fields[9], "dipole_moment_d.z"),
            "total": parse_float(fields[10], "dipole_moment_d.total"),
        },
        "isotropic_polarizability_a0_3": parse_float(fields[11], "isotropic_polarizability_a0_3"),
        "electronic_spatial_extent_au": parse_float(fields[12], "electronic_spatial_extent_au"),
        "zero_point_vibrational_energy_j_mol": parse_float(fields[13], "zero_point_vibrational_energy_j_mol"),
        "zero_point_vibrational_energy_kcal_mol": parse_float(fields[14], "zero_point_vibrational_energy_kcal_mol"),
        TARGET_NAME: parse_float(fields[15], TARGET_NAME),
        "sum_electronic_thermal_energy_ha": parse_float(fields[16], "sum_electronic_thermal_energy_ha"),
        "sum_electronic_thermal_enthalpy_ha": parse_float(fields[17], "sum_electronic_thermal_enthalpy_ha"),
        "sum_electronic_thermal_free_energy_ha": parse_float(fields[18], "sum_electronic_thermal_free_energy_ha"),
        "heat_capacity_j_k": parse_float(fields[19], "heat_capacity_j_k"),
        "orbital_degeneracies": [int(value) for value in fields[20:32]],
        "orbital_symmetries": fields[32:44],
        "subgroup_atom_labels": fields[44],
    }


def parse_xyz(member_name: str, raw: bytes, cell_size: float, archive_name: str) -> tuple[Atoms, dict]:
    text = raw.decode("utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"{member_name}: expected at least 3 non-empty lines.")

    try:
        num_atoms = int(lines[0])
    except ValueError as exc:
        raise ValueError(f"{member_name}: first line is not an atom count: {lines[0]!r}.") from exc

    properties = parse_properties(lines[1])
    atom_lines = lines[2 : 2 + num_atoms]
    if len(atom_lines) != num_atoms:
        raise ValueError(f"{member_name}: expected {num_atoms} atom lines, got {len(atom_lines)}.")

    symbols = []
    positions = []
    mulliken_charges = []
    for line_no, atom_line in enumerate(atom_lines, start=3):
        parts = atom_line.split()
        if len(parts) != 5:
            raise ValueError(f"{member_name}:{line_no}: expected 'symbol x y z charge', got {atom_line!r}.")
        symbols.append(parts[0])
        positions.append([parse_float(parts[idx], f"position[{idx - 1}]") for idx in range(1, 4)])
        mulliken_charges.append(parse_float(parts[4], "mulliken_charge"))

    energy = properties[TARGET_NAME]
    atoms = Atoms(symbols=symbols, positions=positions, cell=[cell_size] * 3, pbc=False)
    atoms.center()
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    atoms.info["source_archive"] = archive_name
    atoms.info["source_member"] = member_name
    atoms.info["qm_sym_properties"] = properties
    atoms.info["target_field_index"] = TARGET_FIELD_INDEX
    atoms.info["target_name"] = TARGET_NAME
    atoms.info["mulliken_charges"] = mulliken_charges

    record = {
        "source_member": member_name,
        "num_atoms": num_atoms,
        "target_energy_ha": energy,
        "properties": properties,
    }
    return atoms, record


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    archive_name = archive_path.name
    if archive_name not in ARCHIVE_CONFIGS:
        allowed = ", ".join(sorted(ARCHIVE_CONFIGS))
        raise SystemExit(f"This converter supports only {allowed}; got {archive_path}.")

    archive_config = ARCHIVE_CONFIGS[archive_name]
    out_path = Path(args.out) if args.out else archive_config["default_out"]
    metadata_path = out_path.with_suffix(".json")
    expected_count = archive_config["expected_count"] if args.expected_count is None else args.expected_count

    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"{out_path} already exists; pass --overwrite to replace it.")
    if metadata_path.exists() and not args.overwrite:
        raise SystemExit(f"{metadata_path} already exists; pass --overwrite to replace it.")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    images = []
    records = []
    with tarfile.open(archive_path, "r") as tar:
        members = sorted(
            (member for member in tar.getmembers() if member.isfile() and member.name.endswith(".xyz")),
            key=lambda member: member.name,
        )
        if expected_count and len(members) != expected_count:
            raise SystemExit(f"Expected {expected_count} XYZ members, found {len(members)}.")

        for member in tqdm(members, desc=f"Converting {archive_name} to ASE", unit=" molecules"):
            member_name = PurePosixPath(member.name).name
            if not member_name.startswith("QM_sym_"):
                raise ValueError(f"Unexpected member name {member.name!r}; expected QM_sym_*.xyz.")
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"Could not read {member.name!r} from {archive_path}.")
            atoms, record = parse_xyz(member_name, handle.read(), args.cell_size, archive_name)
            images.append(atoms)
            records.append(record)

    with out_path.open("wb") as handle:
        pickle.dump(images, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "source_archive": str(archive_path),
        "archive_name": archive_name,
        "num_molecules": len(images),
        "target_field_index": TARGET_FIELD_INDEX,
        "target_name": TARGET_NAME,
        "target_units": "Hartree",
        "cell_size": args.cell_size,
        "pbc": False,
        "records": records,
    }
    with metadata_path.open("w", encoding="utf8") as handle:
        json.dump(metadata, handle, indent=2)

    print(f"Wrote {len(images)} ASE Atoms objects to {out_path}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
