"""
Resource Allocation Agent
=========================
Fixes applied (v3) — based on review:
  ✅ risk_probability (0–1) from GIS Agent is primary signal — no double-counting
  ✅ Formula: 0.60×risk + 0.25×population + 0.15×shortage
  ✅ Continuous population scaling via affected_population when provided
  ✅ infrastructure_vulnerability used as resource multiplier (not in score)
  ✅ Resource inventory tracking — prevents over-allocation beyond what's available
  ✅ Inventory deduction after allocation; can_fulfill() check before deployment
  ✅ Input normalisation: 'high' and 'High' both accepted
"""

import math


class ResourceAllocationAgent:
    """
    Computes a weighted priority score and returns a resource deployment plan.

    Formula (avoids double-counting with GIS inputs):
        score = 0.60 × (risk_probability × 10)   # GIS output encodes sev+pop+infra
              + 0.25 × population_score            # explicit affected count
              + 0.15 × shortage_score              # resource availability

    Infrastructure vulnerability is a MULTIPLIER on final counts — not in score —
    to avoid overlapping with the GIS CPT inputs.
    """

    # ── Base resource tables per disaster type and severity ───────────
    _BASE: dict = {
        'Flood': {
            'Low':    {'ambulances': 2, 'rescue_teams': 1, 'boats': 1, 'fire_trucks': 0},
            'Medium': {'ambulances': 3, 'rescue_teams': 2, 'boats': 3, 'fire_trucks': 0},
            'High':   {'ambulances': 5, 'rescue_teams': 4, 'boats': 5, 'fire_trucks': 0},
        },
        'Earthquake': {
            'Low':    {'ambulances': 2, 'rescue_teams': 2, 'boats': 0, 'fire_trucks': 1},
            'Medium': {'ambulances': 5, 'rescue_teams': 4, 'boats': 0, 'fire_trucks': 2},
            'High':   {'ambulances': 8, 'rescue_teams': 7, 'boats': 0, 'fire_trucks': 4},
        },
        'Landslide': {
            'Low':    {'ambulances': 1, 'rescue_teams': 2, 'boats': 0, 'fire_trucks': 1},
            'Medium': {'ambulances': 3, 'rescue_teams': 3, 'boats': 0, 'fire_trucks': 1},
            'High':   {'ambulances': 5, 'rescue_teams': 5, 'boats': 0, 'fire_trucks': 2},
        },
        'Fire': {
            'Low':    {'ambulances': 1, 'rescue_teams': 1, 'boats': 0, 'fire_trucks': 2},
            'Medium': {'ambulances': 2, 'rescue_teams': 2, 'boats': 0, 'fire_trucks': 4},
            'High':   {'ambulances': 4, 'rescue_teams': 3, 'boats': 0, 'fire_trucks': 6},
        },
        'Tsunami': {
            'Low':    {'ambulances': 3,  'rescue_teams': 4, 'boats': 3, 'fire_trucks': 0},
            'Medium': {'ambulances': 6,  'rescue_teams': 6, 'boats': 5, 'fire_trucks': 0},
            'High':   {'ambulances': 10, 'rescue_teams': 8, 'boats': 8, 'fire_trucks': 0},
        },
        'Cyclone': {
            'Low':    {'ambulances': 3,  'rescue_teams': 3, 'boats': 2, 'fire_trucks': 1},
            'Medium': {'ambulances': 6,  'rescue_teams': 5, 'boats': 4, 'fire_trucks': 2},
            'High':   {'ambulances': 12, 'rescue_teams': 8, 'boats': 6, 'fire_trucks': 3},
        },
    }

    _DEFAULT_BASE = {'ambulances': 2, 'rescue_teams': 2, 'boats': 0, 'fire_trucks': 1}

    # Shortage score: Low availability = high shortage = higher priority
    _SHORTAGE_SCORE: dict = {'High': 2, 'Medium': 5, 'Low': 9}

    # Infrastructure vulnerability → resource count multiplier
    # High vulnerability (weak buildings) → need more rescue / medical
    _INFRA_MULT: dict = {'High': 1.30, 'Medium': 1.00, 'Low': 0.85}

    # Deployment order by disaster type (most critical first)
    _DEPLOY_ORDER: dict = {
        'Flood':      ['boats', 'rescue_teams', 'ambulances', 'fire_trucks'],
        'Earthquake': ['rescue_teams', 'ambulances', 'fire_trucks', 'boats'],
        'Landslide':  ['rescue_teams', 'ambulances', 'fire_trucks', 'boats'],
        'Fire':       ['fire_trucks', 'rescue_teams', 'ambulances', 'boats'],
        'Tsunami':    ['boats', 'rescue_teams', 'ambulances', 'fire_trucks'],
        'Cyclone':    ['rescue_teams', 'boats', 'ambulances', 'fire_trucks'],
    }

    # Default global inventory (units available)
    _DEFAULT_INVENTORY: dict = {
        'ambulances':   10,
        'rescue_teams': 8,
        'boats':        6,
        'fire_trucks':  5,
    }

    # ─────────────────────────────────────────────────────────────────

    def __init__(self, inventory: dict = None):
        """
        Parameters
        ----------
        inventory : dict, optional
            e.g. {'ambulances': 10, 'rescue_teams': 8, 'boats': 6, 'fire_trucks': 5}
            If not provided, _DEFAULT_INVENTORY is used.
        """
        self.inventory: dict = dict(inventory or self._DEFAULT_INVENTORY)

    # ── Main entry point ──────────────────────────────────────────────

    def allocate(
        self,
        incident_type: str,
        severity: str,
        risk_probability: float,
        population_density: str,
        resource_availability: str,
        infrastructure_vulnerability: str = 'Medium',
        affected_population: int = None,
        deduct_from_inventory: bool = True,
    ) -> dict:
        """
        Compute priority score and return resource plan.

        Parameters
        ----------
        risk_probability : float (0–1)
            P(Risk = High) from GISAgent — already encodes sev + pop + infra.
        infrastructure_vulnerability : str
            Used as resource count multiplier, NOT in the score formula.
        affected_population : int, optional
            Absolute number of people at risk — drives population_score.
        deduct_from_inventory : bool
            If True, allocated resources are deducted from inventory.

        Returns
        -------
        dict with:
            priority_score, priority_level, resources, actual_resources,
            inventory_before, inventory_after, deployment_order,
            component_scores, justification
        """
        # Normalise inputs
        itype   = incident_type.title()
        sev     = severity.title()
        pop_d   = population_density.title()
        res_av  = resource_availability.title()
        infra_v = infrastructure_vulnerability.title()

        # Clamp risk_probability to [0, 1]
        risk_prob = max(0.0, min(1.0, float(risk_probability)))

        # ── Score components (no double-counting) ────────────────────
        risk_s = risk_prob * 10.0
        pop_s  = self._population_score(pop_d, affected_population)
        sht_s  = self._SHORTAGE_SCORE.get(res_av, 5)

        score = 0.60 * risk_s + 0.25 * pop_s + 0.15 * sht_s

        # ── Base resources for this disaster + severity ───────────────
        base = (
            self._BASE
            .get(itype, {})
            .get(sev, self._DEFAULT_BASE)
        )

        # ── Score multiplier on resource counts ───────────────────────
        if score >= 7.5:
            score_mult = 1.50
        elif score >= 5.0:
            score_mult = 1.25
        else:
            score_mult = 1.00

        # ── Infrastructure multiplier (separate from score) ───────────
        infra_mult = self._INFRA_MULT.get(infra_v, 1.00)
        combined   = score_mult * infra_mult

        # Required resources (unconstrained by inventory)
        required = {
            k: max(1, math.ceil(v * combined)) if v > 0 else 0
            for k, v in base.items()
        }

        # ── Constrain to available inventory ─────────────────────────
        inventory_before = dict(self.inventory)
        actual = {
            k: min(required[k], self.inventory.get(k, 0))
            for k in required
        }

        # Deduct from inventory if requested
        if deduct_from_inventory:
            for k, v in actual.items():
                self.inventory[k] = max(0, self.inventory.get(k, 0) - v)

        inventory_after = dict(self.inventory)

        # ── Deployment order (skip zero-count types) ──────────────────
        raw_order = self._DEPLOY_ORDER.get(
            itype, ['rescue_teams', 'ambulances', 'fire_trucks', 'boats']
        )
        deployment_order = [r for r in raw_order if actual.get(r, 0) > 0]

        priority_level = self._score_label(score)

        # Warn if any resource was capped by inventory
        shortfall = {k: required[k] - actual[k] for k in required if required[k] > actual[k]}
        shortfall_note = (
            f" ⚠ Shortfall: {shortfall}. Request mutual aid."
            if shortfall else ""
        )

        return {
            'priority_score':    round(score, 2),
            'priority_level':    priority_level,
            'resources':         required,       # what is ideally needed
            'actual_resources':  actual,         # what can actually be deployed
            'inventory_before':  inventory_before,
            'inventory_after':   inventory_after,
            'shortfall':         shortfall,
            'deployment_order':  deployment_order,
            'component_scores': {
                'risk_score (×0.60)':       round(risk_s,    2),
                'population_score (×0.25)': round(pop_s,     2),
                'shortage_score (×0.15)':   round(sht_s,     2),
                'score_multiplier':         round(score_mult, 2),
                'infra_multiplier':         round(infra_mult, 2),
            },
            'justification': (
                f"GIS risk probability: {risk_prob * 100:.1f}% "
                f"(score {risk_s:.1f}/10). "
                f"Population: {pop_d} (score {pop_s}/9). "
                f"Resource shortage: {res_av} (score {sht_s}/9). "
                f"Priority = {score:.2f}/10 → {priority_level}. "
                f"Infra vulnerability = {infra_v} (×{infra_mult:.2f} on resources)."
                f"{shortfall_note}"
            ),
        }

    def restore_inventory(self, resources: dict):
        """Return resources back to inventory (e.g. after incident resolved)."""
        for k, v in resources.items():
            self.inventory[k] = self.inventory.get(k, 0) + v

    def can_fulfill(self, incident_type: str, severity: str) -> bool:
        """Quick check: does inventory have at least 1 unit of each required type?"""
        itype = incident_type.title()
        sev   = severity.title()
        base  = self._BASE.get(itype, {}).get(sev, self._DEFAULT_BASE)
        return all(
            self.inventory.get(k, 0) >= 1
            for k, v in base.items()
            if v > 0
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _population_score(density: str, count: int = None) -> float:
        """
        Continuous score when count is provided; categorical fallback otherwise.
        Scale: 1 person → 1.0,  10,000+ people → 9.0
        """
        if count is not None:
            return min(9.0, max(1.0, 1.0 + (int(count) / 10_000) * 8.0))
        return {'Low': 2.0, 'Medium': 5.0, 'High': 8.0}.get(density, 5.0)

    @staticmethod
    def _score_label(score: float) -> str:
        if score >= 7.5: return 'CRITICAL'
        if score >= 5.0: return 'HIGH'
        if score >= 2.5: return 'MODERATE'
        return 'LOW'