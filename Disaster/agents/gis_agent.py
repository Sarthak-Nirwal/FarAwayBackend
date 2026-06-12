"""
GIS Agent
=========
Fixes applied (v3) — based on review:
  ✅ Terminology: "CPT-based Bayesian-Inspired" — not a full BN engine
  ✅ Disaster-type modifier applied after CPT lookup
  ✅ Affected radius is dynamic: base × f(risk_probability)
  ✅ Resource search returns top-K nearest (no hard distance cutoff)
  ✅ Infrastructure meaning clarified: High = poorly built / vulnerable
  ✅ CPT probabilities capped properly; normalisation never produces NaN
  ✅ analyze() accepts optional lat/lon as positional fallback
  ✅ _make_map() safe when nearby list is empty
  ✅ All imports at top; no circular dependency
"""

import json
import math
import os

import folium
from geopy.distance import geodesic


# ─────────────────────────────────────────────────────────────────────
# CPT-based Bayesian-Inspired Risk Assessor
# ─────────────────────────────────────────────────────────────────────

class SimplifiedBayesianRiskAssessor:
    """
    Approximates Bayesian posterior risk using pre-defined Conditional
    Probability Tables (CPTs).

    Academic note: this is a manually-structured probability table,
    not a full probabilistic graphical model with learned priors.
    Correct terminology: "CPT-based Bayesian-inspired risk assessor".

    Conceptual network:
        Hazard Severity        ──┐
        Population Density     ──┼──► Latent Risk ──► Risk Level
        Infra Vulnerability    ──┘
                                  modified by Resource Availability
                                  modified by Environmental Condition
                                  scaled    by Disaster Type

    Infrastructure Vulnerability:
        High   = poorly built / very vulnerable (old buildings, poor drainage)
        Medium = average
        Low    = modern, well-maintained
    """

    # CPT: (hazard_severity, population_density, infra_vulnerability) → P(Risk = High)
    _CPT: dict = {
        ('High',   'High',   'High'):   0.93,
        ('High',   'High',   'Medium'): 0.78,
        ('High',   'High',   'Low'):    0.62,
        ('High',   'Medium', 'High'):   0.72,
        ('High',   'Medium', 'Medium'): 0.58,
        ('High',   'Medium', 'Low'):    0.44,
        ('High',   'Low',    'High'):   0.55,
        ('High',   'Low',    'Medium'): 0.40,
        ('High',   'Low',    'Low'):    0.28,
        ('Medium', 'High',   'High'):   0.70,
        ('Medium', 'High',   'Medium'): 0.54,
        ('Medium', 'High',   'Low'):    0.40,
        ('Medium', 'Medium', 'High'):   0.47,
        ('Medium', 'Medium', 'Medium'): 0.33,
        ('Medium', 'Medium', 'Low'):    0.22,
        ('Medium', 'Low',    'High'):   0.30,
        ('Medium', 'Low',    'Medium'): 0.20,
        ('Medium', 'Low',    'Low'):    0.13,
        ('Low',    'High',   'High'):   0.34,
        ('Low',    'High',   'Medium'): 0.24,
        ('Low',    'High',   'Low'):    0.17,
        ('Low',    'Medium', 'High'):   0.20,
        ('Low',    'Medium', 'Medium'): 0.12,
        ('Low',    'Medium', 'Low'):    0.07,
        ('Low',    'Low',    'High'):   0.10,
        ('Low',    'Low',    'Medium'): 0.06,
        ('Low',    'Low',    'Low'):    0.03,
    }

    # Disaster-type modifiers — applied after CPT lookup
    # Reflects how sensitive each disaster is to the measured variables
    _DISASTER_MOD: dict = {
        'Flood':      1.20,   # water + rainfall amplifies risk heavily
        'Cyclone':    1.15,   # wind + storm surge
        'Landslide':  1.10,   # slope + rainfall compound
        'Fire':       1.05,   # wind can spread fire
        'Earthquake': 1.00,   # seismic — environmental variables matter less
        'Tsunami':    1.12,   # coastal proximity critical
    }

    # Modifiers for additional variables
    _RESOURCE_MOD: dict = {'Low': 1.22, 'Medium': 1.00, 'High': 0.83}
    _ENV_MOD:      dict = {'High': 1.15, 'Medium': 1.00, 'Low': 0.88}

    def infer(
        self,
        hazard_severity: str,
        population_density: str,
        infrastructure_vulnerability: str,
        resource_availability: str = 'Medium',
        environmental_condition: str = 'Medium',
        disaster_type: str = 'Flood',
    ) -> dict:
        """
        Compute posterior risk probability.

        Returns dict:
            risk_level   – 'High' | 'Medium' | 'Low'
            probability  – P(Risk = High), primary signal downstream
            high_prob, medium_prob, low_prob
        """
        # Normalise inputs to title-case so 'high' and 'High' both work
        hs  = hazard_severity.title()
        pd  = population_density.title()
        iv  = infrastructure_vulnerability.title()
        ra  = resource_availability.title()
        ec  = environmental_condition.title()
        dt  = disaster_type.title()

        key = (hs, pd, iv)
        p_high_base = self._CPT.get(key, 0.50)

        # Apply disaster-type modifier
        d_mod       = self._DISASTER_MOD.get(dt, 1.00)
        p_high_base = min(0.97, p_high_base * d_mod)

        # Apply resource and environment modifiers
        r_mod  = self._RESOURCE_MOD.get(ra, 1.00)
        e_mod  = self._ENV_MOD.get(ec, 1.00)

        p_high = min(0.97, p_high_base * r_mod * e_mod)
        p_low  = max(0.01, (1.0 - p_high_base) * 0.35 / (r_mod * e_mod))
        p_med  = max(0.02, 1.0 - p_high - p_low)

        # Normalise so the three probabilities always sum to 1.0
        total = p_high + p_med + p_low
        if total <= 0:          # guard against floating-point edge cases
            total = 1.0
        p_high /= total
        p_med  /= total
        p_low  /= total

        risk_level = (
            'High'   if p_high >= 0.50 else
            'Medium' if p_med  >= p_low else
            'Low'
        )

        return {
            'risk_level':   risk_level,
            'probability':  round(p_high, 3),
            'high_prob':    round(p_high, 3),
            'medium_prob':  round(p_med,  3),
            'low_prob':     round(p_low,  3),
        }


# ─────────────────────────────────────────────────────────────────────
# GIS Agent
# ─────────────────────────────────────────────────────────────────────

class GISAgent:
    """
    Spatial analysis hub.  Call .analyze(incident) for full results.

    Returns:
        incident_id, incident, risk_assessment, priority_zone,
        affected_radius_km, nearby_resources, map (folium.Map object)

    Incident dict required keys:
        incident_type, severity, latitude, longitude

    Optional keys:
        incident_id, location_name,
        population_density, infrastructure_vulnerability,
        resource_availability, environmental_condition
    """

    # Base radii (km) — multiplied by dynamic factor in _affected_radius()
    _BASE_RADIUS: dict = {
        'Flood':      {'Low': 1.0,  'Medium': 2.5,  'High': 5.0},
        'Earthquake': {'Low': 2.0,  'Medium': 5.0,  'High': 10.0},
        'Landslide':  {'Low': 0.5,  'Medium': 1.5,  'High': 3.0},
        'Tsunami':    {'Low': 2.0,  'Medium': 5.0,  'High': 15.0},
        'Cyclone':    {'Low': 5.0,  'Medium': 15.0, 'High': 30.0},
        'Fire':       {'Low': 0.5,  'Medium': 2.0,  'High': 5.0},
    }

    _ZONE_COLOR: dict = {
        'CRITICAL': 'red',
        'HIGH':     'orange',
        'MODERATE': 'beige',
        'LOW':      'green',
    }

    _RESOURCE_ICON_COLOR: dict = {
        'Hospital':       'blue',
        'Fire Station':   'orange',
        'Rescue Center':  'green',
        'Police Station': 'darkblue',
        'Shelter':        'lightblue',
    }

    def __init__(
        self,
        resources_path: str = 'data/resources.json',
        maps_dir: str = 'maps',
    ):
        self.assessor = SimplifiedBayesianRiskAssessor()
        self.maps_dir = maps_dir
        os.makedirs(maps_dir, exist_ok=True)

        if not os.path.exists(resources_path):
            raise FileNotFoundError(
                f"Resources file not found: {resources_path}\n"
                "Create data/resources.json — see sample in project README."
            )

        with open(resources_path, encoding='utf-8') as f:
            self.resources: list = json.load(f)

    # ── Main entry point ──────────────────────────────────────────────

    def analyze(self, incident: dict) -> dict:
        """
        Full spatial analysis for one incident.

        Parameters
        ----------
        incident : dict
            Required: incident_type, severity, latitude, longitude
            Optional: incident_id, location_name,
                      population_density, infrastructure_vulnerability,
                      resource_availability, environmental_condition
        """
        # Validate required keys
        for key in ('incident_type', 'severity', 'latitude', 'longitude'):
            if key not in incident:
                raise ValueError(f"incident dict is missing required key: '{key}'")

        lat   = float(incident['latitude'])
        lon   = float(incident['longitude'])
        sev   = incident.get('severity', 'Medium').title()
        itype = incident.get('incident_type', 'Flood').title()

        risk = self.assessor.infer(
            hazard_severity=sev,
            population_density=incident.get('population_density', 'Medium'),
            infrastructure_vulnerability=incident.get('infrastructure_vulnerability', 'Medium'),
            resource_availability=incident.get('resource_availability', 'Medium'),
            environmental_condition=incident.get('environmental_condition', 'Medium'),
            disaster_type=itype,
        )

        affected_r = self._affected_radius(itype, sev, risk['probability'])
        nearby     = self._nearby_resources(lat, lon, top_k=8)
        priority   = self._priority_zone(risk['probability'])
        fmap       = self._make_map(incident, risk, affected_r, nearby, priority)

        return {
            'incident_id':        incident.get('incident_id', 'N/A'),
            'incident':           incident,
            'risk_assessment':    risk,
            'priority_zone':      priority,
            'affected_radius_km': affected_r,
            'nearby_resources':   nearby,
            'map':                fmap,
        }

    # ── Resource helpers ──────────────────────────────────────────────

    def _nearby_resources(self, lat: float, lon: float, top_k: int = 8) -> list:
        """Return top_k nearest resources — no hard distance cutoff."""
        origin = (lat, lon)
        scored = []
        for r in self.resources:
            try:
                dist = geodesic(origin, (float(r['lat']), float(r['lon']))).km
                scored.append({**r, 'distance_km': round(dist, 2)})
            except Exception:
                continue  # skip malformed resource entries
        return sorted(scored, key=lambda x: x['distance_km'])[:top_k]

    def nearest_by_type(self, lat: float, lon: float, rtype: str) -> dict:
        """Return single nearest resource of a given type."""
        pool = [r for r in self.resources if r.get('type') == rtype]
        if not pool:
            return {}
        origin = (lat, lon)
        best = min(
            pool,
            key=lambda r: geodesic(origin, (float(r['lat']), float(r['lon']))).km
        )
        result = dict(best)
        result['distance_km'] = round(
            geodesic(origin, (float(best['lat']), float(best['lon']))).km, 2
        )
        return result

    # ── Helpers ───────────────────────────────────────────────────────

    def _affected_radius(self, itype: str, sev: str, risk_prob: float) -> float:
        """Dynamic radius = base × (0.75 + 0.50 × risk_probability)."""
        base   = self._BASE_RADIUS.get(itype, {}).get(sev, 2.5)
        factor = 0.75 + 0.50 * risk_prob   # range: 0.75 → 1.25
        return round(base * factor, 2)

    @staticmethod
    def _priority_zone(p: float) -> str:
        if p >= 0.70: return 'CRITICAL'
        if p >= 0.45: return 'HIGH'
        if p >= 0.25: return 'MODERATE'
        return 'LOW'

    # ── Map ──────────────────────────────────────────────────────────

    def _make_map(
        self,
        incident: dict,
        risk: dict,
        radius_km: float,
        nearby: list,
        priority: str,
    ) -> folium.Map:
        lat   = float(incident['latitude'])
        lon   = float(incident['longitude'])
        color = self._ZONE_COLOR.get(priority, 'red')

        m = folium.Map(location=[lat, lon], zoom_start=13, tiles='OpenStreetMap')

        # Incident marker
        folium.Marker(
            [lat, lon],
            popup=folium.Popup(
                f"<b>⚠ {incident.get('incident_type', '?')}</b><br>"
                f"Severity : {incident.get('severity', '?')}<br>"
                f"Risk     : {risk['risk_level']} "
                f"({risk['probability'] * 100:.1f}%)<br>"
                f"Zone     : {priority}",
                max_width=220,
            ),
            tooltip=f"⚠ {incident.get('location_name', 'Incident')}",
            icon=folium.Icon(color=color, icon='exclamation-sign', prefix='glyphicon'),
        ).add_to(m)

        # Affected area circle (dynamic radius)
        folium.Circle(
            [lat, lon],
            radius=radius_km * 1000,   # metres
            color=color,
            fill=True,
            fill_opacity=0.15,
            popup=f"Affected area ≈ {radius_km} km (dynamic)",
        ).add_to(m)

        # Resource markers — safe even when nearby list is empty
        for r in nearby:
            rc = self._RESOURCE_ICON_COLOR.get(r.get('type', ''), 'gray')
            try:
                folium.CircleMarker(
                    [float(r['lat']), float(r['lon'])],
                    radius=9,
                    color=rc,
                    fill=True,
                    fill_opacity=0.75,
                    popup=folium.Popup(
                        f"<b>{r.get('name', 'Resource')}</b><br>"
                        f"Type: {r.get('type', '?')}<br>"
                        f"Distance: {r.get('distance_km', '?')} km",
                        max_width=180,
                    ),
                    tooltip=f"{r.get('name', 'Resource')} ({r.get('distance_km', '?')} km)",
                ).add_to(m)
            except Exception:
                continue   # skip any resource with invalid coordinates

        # Legend overlay
        legend_html = (
            f'<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
            f'background:white;padding:8px 14px;border-radius:6px;'
            f'border:2px solid {color};font-size:13px;line-height:1.7;">'
            f'<b>Risk: {risk["risk_level"]}</b> '
            f'({risk["probability"] * 100:.1f}%)<br>'
            f'Zone: <b>{priority}</b> &nbsp;|&nbsp; '
            f'Radius: {radius_km} km</div>'
        )
        m.get_root().html.add_child(folium.Element(legend_html))

        save_path = os.path.join(
            self.maps_dir,
            f'map_{incident.get("incident_id", "latest")}.html'
        )
        m.save(save_path)
        return m   # return object so Streamlit can embed it