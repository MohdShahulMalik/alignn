from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import dgl
import pandas as pd
import torch
from torch.utils.data import Dataset

from alignn.data.dgl_graph import CrystalGraph, build_dgl_graph
from alignn.data.graph_builder import build_atom_graph
from alignn.data.structure import jarvis_atoms_to_structure


@dataclass
class GraphTargetSample:
    crystal: CrystalGraph
    target: torch.Tensor
    jid: str


def _resolve_raw_archive(project_root: Path, dataset_name: str) -> Path:
    raw_dir = project_root / "data" / "raw"
    patterns = [f"{dataset_name}*.json.zip", f"{dataset_name}*.zip"]
    if dataset_name == "dft_3d":
        patterns = ["jdft_3d*.json.zip", "jdft_3d*.zip", *patterns]

    for pattern in patterns:
        matches = sorted(raw_dir.glob(pattern))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"No raw archive found for dataset '{dataset_name}' in {raw_dir}."
    )


def _load_raw_records(raw_archive: Path) -> dict[str, dict]:
    with zipfile.ZipFile(raw_archive) as zf:
        json_name = zf.namelist()[0]
        with zf.open(json_name) as handle:
            records = json.load(handle)
    return {record["jid"]: record for record in records if "jid" in record}


class JarvisGraphDataset(Dataset[GraphTargetSample]):
    """Build crystal graphs from prepared JARVIS split files."""

    def __init__(
        self,
        project_root: Path,
        split: str,
        dataset_name: str = "dft_3d",
        target_column: str = "formation_energy_peratom",
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        compute_line_graph: bool = False,
        cache_graphs: bool = True,
    ) -> None:
        self.project_root = project_root.resolve()
        self.split = split
        self.dataset_name = dataset_name
        self.target_column = target_column
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.compute_line_graph = compute_line_graph
        self.cache_graphs = cache_graphs

        split_path = (
            self.project_root
            / "data"
            / "splits"
            / f"{dataset_name}_{target_column}_{split}.csv"
        )
        if not split_path.exists():
            raise FileNotFoundError(f"Missing split file: {split_path}")

        self.frame = pd.read_csv(split_path)
        self.records_by_jid = _load_raw_records(
            _resolve_raw_archive(self.project_root, dataset_name)
        )
        self.graph_cache: dict[str, CrystalGraph] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _build_sample_graph(self, jid: str) -> CrystalGraph:
        record = self.records_by_jid.get(jid)
        if record is None:
            raise KeyError(f"JID '{jid}' not found in raw archive.")

        structure = jarvis_atoms_to_structure(record["atoms"])
        graph_dict = build_atom_graph(
            structure=structure,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
        )
        return build_dgl_graph(
            graph_dict=graph_dict,
            compute_line_graph=self.compute_line_graph,
        )

    def __getitem__(self, index: int) -> GraphTargetSample:
        row = self.frame.iloc[index]
        jid = str(row["jid"])

        crystal = self.graph_cache.get(jid)
        if crystal is None:
            crystal = self._build_sample_graph(jid)
            if self.cache_graphs:
                self.graph_cache[jid] = crystal

        return GraphTargetSample(
            crystal=crystal,
            target=torch.tensor(float(row["target"]), dtype=torch.float32),
            jid=jid,
        )


def collate_graph_samples(
    samples: list[GraphTargetSample],
) -> tuple[dgl.DGLGraph, torch.Tensor, list[str]]:
    graphs = [sample.crystal.g for sample in samples]
    targets = torch.stack([sample.target for sample in samples])
    jids = [sample.jid for sample in samples]
    return dgl.batch(graphs), targets, jids


def collate_graph_samples_with_line_graph(
    samples: list[GraphTargetSample],
) -> tuple[dgl.DGLGraph, dgl.DGLGraph, torch.Tensor, list[str]]:
    graphs = [sample.crystal.g for sample in samples]
    line_graphs = [sample.crystal.lg for sample in samples]
    targets = torch.stack([sample.target for sample in samples])
    jids = [sample.jid for sample in samples]
    return dgl.batch(graphs), dgl.batch(line_graphs), targets, jids
