from pathlib import Path

import pytest

from src.profile_search import build_queries, load_profiles


def test_load_profiles_creates_editable_default_file(tmp_path: Path):
    profile_path = tmp_path / "kb_profiles.json"

    profiles = load_profiles(profile_path)

    assert profile_path.exists()
    assert "outreach" in profiles
    assert profiles["outreach"][0]


def test_build_queries_appends_context_to_each_profile_query(tmp_path: Path):
    profile_path = tmp_path / "kb_profiles.json"
    profiles = load_profiles(profile_path)

    queries = build_queries("outreach", "ReVesta surplus recovery firms", profiles)

    assert len(queries) >= 3
    assert all("ReVesta surplus recovery firms" in query for query in queries)


def test_build_queries_accepts_custom_query_profile_name(tmp_path: Path):
    profiles = load_profiles(tmp_path / "kb_profiles.json")

    queries = build_queries("q", "pricing guarantee for B2B data product", profiles)

    assert queries == ["pricing guarantee for B2B data product"]


def test_build_queries_rejects_unknown_profile(tmp_path: Path):
    profiles = load_profiles(tmp_path / "kb_profiles.json")

    with pytest.raises(KeyError):
        build_queries("missing-profile", "context", profiles)
