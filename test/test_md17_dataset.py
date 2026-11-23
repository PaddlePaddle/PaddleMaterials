import numpy as np

from ppmat.datasets.collate_fn import RadiusGraphCollator
from ppmat.datasets.custom_data_type import ConcatData
from ppmat.datasets.md17_dataset import MD17Dataset


def _write_md17_data(data_dir):
    data_dir.mkdir(parents=True)
    split_dir = data_dir / "splits"
    split_dir.mkdir()

    atomic_numbers = np.array([6, 1, 8, 1], dtype=np.int32)
    base_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [0.0, 0.0, 1.2],
        ],
        dtype=np.float64,
    )
    positions = np.stack(
        [base_positions + frame * 0.01 for frame in range(4)]
    )
    energies = np.array([-1.0, -2.0, -3.0, -4.0], dtype=np.float64)
    forces = np.arange(4 * 4 * 3, dtype=np.float64).reshape(4, 4, 3)

    data_path = data_dir / "md17_aspirin.npz"
    np.savez(
        data_path,
        z=atomic_numbers,
        R=positions,
        E=energies,
        F=forces,
    )
    np.save(split_dir / "aspirin_train_idx.npy", np.array([3, 1]))
    return data_path, forces


def _dataset_config():
    return {
        "build_molecule_cfg": {
            "format": "dict",
            "sanitize": False,
            "add_hs": False,
            "remove_hs": False,
            "kekulize": False,
            "num_cpus": 1,
        },
        "build_graph_cfg": {
            "__class_name__": "RadiusGraphConverter",
            "__init_params__": {
                "cutoff": 5.0,
                "return_triplet_indices": True,
                "num_cpus": 1,
            },
        },
    }


def test_md17_split_cache_and_collation(tmp_path):
    data_path, forces = _write_md17_data(tmp_path / "md17")
    dataset = MD17Dataset(
        path=str(data_path),
        name="aspirin",
        split="train",
        cache_path=str(tmp_path / "cache"),
        **_dataset_config(),
    )

    assert len(dataset) == 2
    assert dataset.sample_ids.tolist() == [3, 1]
    assert dataset.raw_data["z"].dtype == np.int64
    assert dataset.raw_data["pos"].dtype == np.float32
    assert dataset.property_data["energy"].dtype == np.float32
    assert dataset.property_data["force"].dtype == np.float32

    sample = dataset[0]
    assert set(sample) == {"graph", "energy", "force", "id"}
    assert sample["id"] == 3
    np.testing.assert_array_equal(sample["energy"], [-4.0])
    assert isinstance(sample["force"], ConcatData)
    np.testing.assert_array_equal(sample["force"].data, forces[3].astype("float32"))

    samples = [dataset[0], dataset[1]]
    batch = RadiusGraphCollator()(samples)
    assert set(batch) == {"graph", "energy", "force", "id"}
    assert batch["energy"].shape == (2, 1)
    assert batch["force"].shape == (8, 3)
    np.testing.assert_array_equal(batch["id"], [3, 1])
    assert int(batch["graph"].num_nodes) == 8
    first_num_edges = np.asarray(samples[0]["graph"].edges).shape[0]
    for key in ("ti_idx_kj", "ti_idx_ji"):
        assert key in batch["graph"].edge_feat
        expected = np.concatenate(
            [
                np.asarray(samples[0]["graph"].edge_feat[key]),
                np.asarray(samples[1]["graph"].edge_feat[key])
                + first_num_edges,
            ]
        )
        np.testing.assert_array_equal(batch["graph"].edge_feat[key], expected)


def test_md17_download_path_uses_extracted_directory(tmp_path, monkeypatch):
    extract_root = tmp_path / "download"
    data_path, _ = _write_md17_data(extract_root / "md17")
    requested_path = tmp_path / "missing" / data_path.name

    monkeypatch.setattr(
        "ppmat.datasets.md17_dataset.download.get_datasets_path_from_url",
        lambda url, md5: str(extract_root),
    )
    dataset = MD17Dataset(
        path=str(requested_path),
        name="aspirin",
        split="train",
        cache_path=str(tmp_path / "download_cache"),
        **_dataset_config(),
    )

    assert dataset.path == str(data_path)
    assert len(dataset) == 2
