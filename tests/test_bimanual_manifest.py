"""Bimanual workstation manifest sanity tests.

Pure-Python: loads the composed `ar5_l6_bench_bimanual` manifest
through `sim.registry.load` and checks the shape (role set, joint
counts per role, per-role ee_links, gain_profiles per role). Does
not import Isaac — these assertions guard the composer + registry
contract, not any simulator.

Expected to be kept green by re-running the composer whenever a
component meta or the bimanual recipe changes.
"""

from __future__ import annotations

from sim.registry import load


def test_bimanual_workstation_has_five_roles():
    h = load("ar5_l6_bench_bimanual")
    assert set(h.joints) == {
        "base", "arm_left", "arm_right", "hand_left", "hand_right"
    }


def test_bimanual_joint_counts_per_role():
    h = load("ar5_l6_bench_bimanual")
    assert len(h.joints["base"]) == 0
    assert len(h.joints["arm_left"]) == 7
    assert len(h.joints["arm_right"]) == 7
    assert len(h.joints["hand_left"]) == 6
    assert len(h.joints["hand_right"]) == 6
    # Mimic joints stay per-role (5 per hand, none for arms).
    assert len(h.mimic_joints["hand_left"]) == 5
    assert len(h.mimic_joints["hand_right"]) == 5
    assert len(h.mimic_joints["arm_left"]) == 0


def test_bimanual_per_role_ee_links_differ():
    h = load("ar5_l6_bench_bimanual")
    assert set(h.ee_links) == {"arm_left", "arm_right"}
    assert h.ee_links["arm_left"] != h.ee_links["arm_right"]
    # ee_link (singular) stays for back-compat and mirrors the first arm.
    assert h.ee_link == h.ee_links["arm_left"]


def test_bimanual_gain_profiles_keyed_by_role():
    h = load("ar5_l6_bench_bimanual")
    # Every arm role carries the composed `osc` + `joint` profiles
    # from arms/ar5/meta.yaml.
    for role in ("arm_left", "arm_right"):
        profiles = h.gain_profiles[role]
        assert "osc" in profiles and "joint" in profiles and "default" in profiles
        assert profiles["osc"].stiffness == 150.0
        assert profiles["osc"].damping == 8.0
    # Hand roles carry only `joint` + `default` (no osc profile in meta).
    for role in ("hand_left", "hand_right"):
        profiles = h.gain_profiles[role]
        assert "joint" in profiles and "default" in profiles


def test_bimanual_frames_expose_both_tool0():
    h = load("ar5_l6_bench_bimanual")
    assert "arm_left:tool0" in h.frames
    assert "arm_right:tool0" in h.frames
    assert h.frames["arm_left:tool0"] != h.frames["arm_right:tool0"]
