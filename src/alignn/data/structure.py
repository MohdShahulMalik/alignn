from __future__ import annotations

from jarvis.core.atoms import Atoms
from pymatgen.core import Structure

def jarvis_atoms_to_structure(atoms_dict: dict) -> Structure:
    """
    Convert a Jarvis Atoms object to a Pymatgen Structure object.

    Args:
        jarvis_atoms (Atoms): A Jarvis Atoms object.

    Returns:
        Structure: A Pymatgen Structure object.
    """
    jarvis_atoms = Atoms.from_dict(atoms_dict)
    return jarvis_atoms.pymatgen_converter()
