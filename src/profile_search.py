"""Editable search profiles for the Founder KB helper CLI."""

from __future__ import annotations

import json
from pathlib import Path

Profiles = dict[str, list[str]]

DEFAULT_PROFILES: Profiles = {
    "outreach": [
        "cold outreach lead magnet demonstrate big value fast",
        "B2B outbound offer follow-up getting leads",
        "make offer so good people feel stupid saying no",
    ],
    "calls": [
        "Never Split the Difference tactical empathy labels calibrated questions sales",
        "customer discovery interview questions problem pain buying process",
        "sales call objection handling founder",
    ],
    "discovery": [
        "The Mom Test customer discovery bad questions good questions",
        "turn hypotheses into facts get out of the building",
        "startup validation interview problem workflow budget",
    ],
    "offer": [
        "100M Offers value equation guarantee risk reversal pricing",
        "offer creation value stack dream outcome likelihood effort delay",
        "pricing psychology guarantee founder sales",
    ],
    "positioning": [
        "Obviously Awesome positioning competitive alternatives unique value customer segment",
        "Zero to One monopoly secrets competition differentiation",
        "category design product positioning startup",
    ],
    "gtm": [
        "Traction channels bullseye framework test channels",
        "lead generation growth channels cold outreach content paid ads",
        "startup go to market first customers",
    ],
    "revesta": [
        "B2B lead generation cold outreach offer lead magnet",
        "customer discovery B2B buying process problem interviews",
        "positioning niche market premium data product",
        "pricing ROI guarantee B2B data product",
    ],
}


def load_profiles(profile_path: Path) -> Profiles:
    """Load editable profiles, creating a default JSON file when missing."""
    if not profile_path.exists():
        profile_path.write_text(
            json.dumps(DEFAULT_PROFILES, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return dict(DEFAULT_PROFILES)

    data = json.loads(profile_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Profile file must contain an object: {profile_path}")

    profiles: Profiles = {}
    for name, queries in data.items():
        if not isinstance(name, str) or not isinstance(queries, list):
            raise ValueError(f"Invalid profile entry for {name!r}")
        normalized = [str(query).strip() for query in queries if str(query).strip()]
        if normalized:
            profiles[name] = normalized
    return profiles


def build_queries(profile: str, context: str, profiles: Profiles) -> list[str]:
    """Build concrete queries for a profile plus optional task context."""
    clean_context = context.strip()
    if profile in {"q", "query", "search"}:
        if not clean_context:
            raise ValueError("Custom query requires text")
        return [clean_context]

    if profile not in profiles:
        raise KeyError(profile)

    if not clean_context:
        return profiles[profile]

    return [f"{query} | context: {clean_context}" for query in profiles[profile]]
