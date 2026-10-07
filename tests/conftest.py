import pytest


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    """Point the pipeline's upload dir and persisted-store paths at a temp dir.

    Returns the (already created) upload directory. Modules bind these names
    with `from config import ...`, so only `src.pipeline.*` needs patching.
    """
    upload_dir = tmp_path / "uploads"
    store_dir = tmp_path / "vector_store"
    upload_dir.mkdir()
    store_dir.mkdir()
    monkeypatch.setattr("src.pipeline.UPLOAD_DIR", upload_dir)
    monkeypatch.setattr("src.pipeline.FAISS_INDEX_PATH", store_dir / "index.faiss")
    monkeypatch.setattr("src.pipeline.METADATA_STORE_PATH", store_dir / "metadata.json")
    monkeypatch.setattr("src.pipeline.DOCUMENT_REGISTRY_PATH", store_dir / "documents.json")
    return upload_dir
