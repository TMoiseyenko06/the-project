from __future__ import annotations

import random
from pathlib import Path

import pytest

from imagebatch.faces import DEFAULT_MATCH_THRESHOLD, Detection, FaceRegistry
from imagebatch.storage import StorageError


def unit_vector(seed: int, dims: int = 128) -> list[float]:
    """A deterministic, roughly-unit-length synthetic embedding.

    128 dims (real ArcFace embeddings are 512-d) so that two independently
    random "different faces" land reliably far apart in cosine similarity —
    at low dimensionality random vectors collide by chance too often to be a
    reliable stand-in for genuinely different faces.
    """
    rng = random.Random(seed)
    v = [rng.uniform(-1, 1) for _ in range(dims)]
    norm = sum(x * x for x in v) ** 0.5
    return [x / norm for x in v]


def nudged(vec: list[float], amount: float, seed: int) -> list[float]:
    """A small perturbation of *vec* — simulates the same face from a different photo."""
    rng = random.Random(seed)
    return [x + rng.uniform(-amount, amount) for x in vec]


@pytest.fixture
def registry(tmp_path: Path) -> FaceRegistry:
    return FaceRegistry(tmp_path / "faces.json")


# -- matching / identity --------------------------------------------------
def test_first_face_creates_new_id(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    assert face_id == "face_001"
    assert registry.get(face_id).sample_count == 1


def test_similar_embedding_matches_existing_face(registry: FaceRegistry) -> None:
    base = unit_vector(1)
    first = registry.match_or_create(base)

    second = registry.match_or_create(nudged(base, 0.02, seed=2))

    assert second == first
    assert registry.get(first).sample_count == 2


def test_dissimilar_embedding_creates_new_face(registry: FaceRegistry) -> None:
    first = registry.match_or_create(unit_vector(1))
    second = registry.match_or_create(unit_vector(999))  # unrelated random vector

    assert second != first
    assert len(registry.list_faces()) == 2


def test_three_photos_of_two_people_cluster_correctly(registry: FaceRegistry) -> None:
    alex = unit_vector(10)
    jordan = unit_vector(20)

    a1 = registry.match_or_create(nudged(alex, 0.01, seed=1))
    j1 = registry.match_or_create(nudged(jordan, 0.01, seed=2))
    a2 = registry.match_or_create(nudged(alex, 0.01, seed=3))

    assert a1 == a2
    assert a1 != j1
    assert len(registry.list_faces()) == 2


def test_sample_count_increments_on_each_match(registry: FaceRegistry) -> None:
    base = unit_vector(1)
    face_id = registry.match_or_create(base)
    assert registry.get(face_id).sample_count == 1

    registry.match_or_create(nudged(base, 0.02, seed=2))
    assert registry.get(face_id).sample_count == 2

    registry.match_or_create(nudged(base, 0.02, seed=3))
    assert registry.get(face_id).sample_count == 3


def test_ids_are_sequential_and_zero_padded(registry: FaceRegistry) -> None:
    ids = [registry.match_or_create(unit_vector(i)) for i in range(1, 12)]
    assert ids[0] == "face_001"
    assert ids[10] == "face_011"


def test_threshold_is_configurable(tmp_path: Path) -> None:
    strict = FaceRegistry(tmp_path / "strict.json", match_threshold=0.999)
    base = unit_vector(1)
    first = strict.match_or_create(base)
    # Even a tiny nudge should fail such a strict threshold and create a new face.
    second = strict.match_or_create(nudged(base, 0.05, seed=2))
    assert first != second


# -- new face registration defaults ----------------------------------------
def test_new_face_has_no_name(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    face = registry.get(face_id)
    assert face.name is None
    assert face.label == face_id  # falls back to the id until named


# -- renaming ---------------------------------------------------------------
def test_rename_sets_label(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    registry.rename(face_id, "Alex")
    face = registry.get(face_id)
    assert face.name == "Alex"
    assert face.label == "Alex"


def test_rename_to_empty_clears_name(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    registry.rename(face_id, "Alex")
    registry.rename(face_id, "  ")
    assert registry.get(face_id).name is None


def test_rename_missing_face_rejected(registry: FaceRegistry) -> None:
    with pytest.raises(StorageError, match="no such face"):
        registry.rename("face_999", "Alex")


def test_rename_does_not_change_face_id(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    registry.rename(face_id, "Alex")
    # a future match against the same embedding still returns the same stable id
    same = registry.match_or_create(unit_vector(1))
    assert same == face_id


# -- merging (fixing a split) -----------------------------------------------
def test_merge_combines_sample_counts(registry: FaceRegistry) -> None:
    base = unit_vector(1)
    keep = registry.match_or_create(base)
    registry.match_or_create(nudged(base, 0.02, seed=2))  # sample_count=2 on keep
    other = registry.match_or_create(unit_vector(500))  # a separate, wrongly-split face

    merged = registry.merge(keep, other)

    assert merged.face_id == keep
    assert merged.sample_count == 3  # 2 + 1
    with pytest.raises(StorageError, match="no such face"):
        registry.get(other)


def test_merge_keeps_existing_name(registry: FaceRegistry) -> None:
    keep = registry.match_or_create(unit_vector(1))
    other = registry.match_or_create(unit_vector(500))
    registry.rename(keep, "Alex")
    registry.rename(other, "Alexander")

    merged = registry.merge(keep, other)

    assert merged.name == "Alex"  # keep_id's name wins when both are set


def test_merge_adopts_name_if_keep_unnamed(registry: FaceRegistry) -> None:
    keep = registry.match_or_create(unit_vector(1))
    other = registry.match_or_create(unit_vector(500))
    registry.rename(other, "Alex")

    merged = registry.merge(keep, other)

    assert merged.name == "Alex"


def test_merge_self_rejected(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    with pytest.raises(StorageError, match="itself"):
        registry.merge(face_id, face_id)


def test_merge_missing_face_rejected(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    with pytest.raises(StorageError, match="no such face"):
        registry.merge(face_id, "face_999")


def test_merged_embedding_enables_future_matches(registry: FaceRegistry) -> None:
    """After merging, a new photo of that person should still match the survivor."""
    base = unit_vector(1)
    keep = registry.match_or_create(base)
    other = registry.match_or_create(unit_vector(500))
    registry.merge(keep, other)

    again = registry.match_or_create(nudged(base, 0.02, seed=7))

    assert again == keep


# -- deletion -----------------------------------------------------------
def test_delete_face(registry: FaceRegistry) -> None:
    face_id = registry.match_or_create(unit_vector(1))
    registry.delete(face_id)
    assert registry.list_faces() == []


def test_delete_missing_face_rejected(registry: FaceRegistry) -> None:
    with pytest.raises(StorageError, match="no such face"):
        registry.delete("face_999")


def test_delete_then_rematch_creates_fresh_id(registry: FaceRegistry) -> None:
    base = unit_vector(1)
    first = registry.match_or_create(base)
    registry.delete(first)

    second = registry.match_or_create(base)

    assert second != first  # the old identity is genuinely gone, not just hidden


# -- listing --------------------------------------------------------------
def test_list_faces_sorted_by_label(registry: FaceRegistry) -> None:
    a = registry.match_or_create(unit_vector(1))
    b = registry.match_or_create(unit_vector(2))
    registry.rename(a, "Zeta")
    registry.rename(b, "Alpha")

    labels = [f.label for f in registry.list_faces()]

    assert labels == ["Alpha", "Zeta"]


def test_unnamed_faces_sort_by_id(registry: FaceRegistry) -> None:
    registry.match_or_create(unit_vector(1))
    registry.match_or_create(unit_vector(2))
    labels = [f.label for f in registry.list_faces()]
    assert labels == ["face_001", "face_002"]


# -- persistence ------------------------------------------------------------
def test_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "faces.json"
    reg = FaceRegistry(path)
    base = unit_vector(1)
    face_id = reg.match_or_create(base)
    reg.rename(face_id, "Alex")

    reopened = FaceRegistry(path)

    assert reopened.get(face_id).name == "Alex"
    # matching still works after reload, not just plain lookups
    assert reopened.match_or_create(nudged(base, 0.02, seed=9)) == face_id


def test_corrupt_file_raises_clear_error(tmp_path: Path) -> None:
    path = tmp_path / "faces.json"
    path.write_text("{ not json")
    with pytest.raises(StorageError, match="corrupt"):
        FaceRegistry(path).list_faces()


# -- Detection dataclass ----------------------------------------------------
def test_detection_is_a_plain_dataclass() -> None:
    d = Detection(embedding=[0.1, 0.2], bbox=(0.0, 0.0, 10.0, 10.0))
    assert d.embedding == [0.1, 0.2]
    assert d.bbox == (0.0, 0.0, 10.0, 10.0)


def test_default_threshold_is_reasonable() -> None:
    assert 0.0 < DEFAULT_MATCH_THRESHOLD < 1.0
