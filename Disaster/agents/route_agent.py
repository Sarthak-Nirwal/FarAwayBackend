"""
Route Optimization Agent
========================
Fixes applied (v3) — based on review:
  ✅ Blocked roads REMOVED from graph copy — not just penalised with weight 9999
  ✅ nearest_node() uses haversine (true km) — not Euclidean degree math
  ✅ Edge distances auto-computed from node coordinates (no hand-coded km)
  ✅ Composite cost: time_minutes × traffic × (1 + flood_risk×3 + road_damage×2)
     — now minimises TRAVEL TIME, not just distance (fixes ETA optimisation bug)
  ✅ update_road_risks() — GIS Agent feeds flood/damage data into edge weights
  ✅ find_best_resource_route() — tries multiple resources, picks fastest ETA
  ✅ A* heuristic is time-based (dist/max_speed) for admissibility
  ✅ Road risks decay over time via decay_road_risks()
  ✅ Input normalisation on node names (lowercase strip)
"""

import math
import time
import networkx as nx


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km (Haversine formula)."""
    R    = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dp   = math.radians(lat2 - lat1)
    dl   = math.radians(lon2 - lon1)
    a    = (math.sin(dp / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2)
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


class RouteOptimizationAgent:
    """
    A* route optimisation on a road graph.

    Usage
    -----
        agent = RouteOptimizationAgent()

        # Simple A-to-B route
        result = agent.find_route('doon_hospital', 'parade_ground')

        # Auto-snap incident coordinates → nearest node → route
        result = agent.route_to_incident('ndrf_camp', 30.3165, 78.0322)

        # Update road risks from GIS flood analysis
        agent.update_road_risks({'Rajpur Rd': {'flood_risk': 0.8, 'road_damage': 0.4}})

        # Pick fastest route from several available resources
        result = agent.find_best_resource_route(
            ['doon_hospital', 'ndrf_camp'], 'parade_ground'
        )

        # Decay risks after time passes
        agent.decay_road_risks(decay_factor=0.85)
    """

    # Speeds (km/h) by road type — used for time calculation and heuristic
    _SPEED: dict = {'highway': 60, 'city': 30, 'residential': 20, 'track': 15}
    _MAX_SPEED = 60.0    # used in admissible heuristic

    def __init__(self):
        self.nodes: dict = {}       # name → (lat, lon)
        self._risk_timestamps: dict = {}   # road_name → unix timestamp of last update
        self.graph = self._build_network()

    # ─────────────────────────────────────────────────────────────────
    # Network construction
    # ─────────────────────────────────────────────────────────────────

    def _build_network(self) -> nx.Graph:
        G = nx.Graph()

        # Key node positions — (lat, lon) for Dehradun area
        self.nodes = {
            'doon_hospital':       (30.3255, 78.0421),
            'coronation_hospital': (30.3162, 78.0322),
            'max_hospital':        (30.3408, 78.0648),
            'ndrf_camp':           (30.3445, 77.9853),
            'prem_nagar':          (30.3526, 77.9857),
            'clock_tower':         (30.3245, 78.0417),
            'parade_ground':       (30.3212, 78.0380),
            'railway_station':     (30.3166, 78.0332),
            'govindgarh':          (30.3158, 78.0276),
            'ballupur_chowk':      (30.3408, 78.0524),
            'rajpur_road_top':     (30.3580, 78.0680),
            'race_course':         (30.3360, 78.0380),
            'rest_camp':           (30.3155, 78.0510),
            'haridwar_road':       (30.2983, 78.0462),
            'sector_17':           (30.3025, 78.0322),
            'nehru_colony':        (30.3028, 78.0445),
            'ec_road_mid':         (30.3185, 78.0465),
        }

        for name, (lat, lon) in self.nodes.items():
            G.add_node(name, lat=lat, lon=lon)

        # Roads: (from, to, road_name, road_type)
        # Distances auto-computed from haversine coordinates
        roads = [
            ('doon_hospital',      'clock_tower',        'Subhash Rd',       'city'),
            ('clock_tower',        'parade_ground',      'Gandhi Rd',        'city'),
            ('parade_ground',      'railway_station',    'Station Rd',       'city'),
            ('railway_station',    'govindgarh',         'Station Rd',       'city'),
            ('govindgarh',         'coronation_hospital','Coronation Rd',    'residential'),
            ('govindgarh',         'sector_17',          'Ring Rd East',     'city'),
            ('clock_tower',        'ballupur_chowk',     'Rajpur Rd',        'city'),
            ('ballupur_chowk',     'max_hospital',       'GMS Rd',           'city'),
            ('ballupur_chowk',     'rajpur_road_top',    'Rajpur Rd',        'city'),
            ('ballupur_chowk',     'race_course',        'EC Rd',            'city'),
            ('race_course',        'parade_ground',      'EC Rd',            'city'),
            ('parade_ground',      'ec_road_mid',        'EC Rd',            'city'),
            ('ec_road_mid',        'rest_camp',          'EC Rd',            'city'),
            ('clock_tower',        'rest_camp',          'Haridwar Bypass',  'highway'),
            ('rest_camp',          'haridwar_road',      'Haridwar Rd',      'highway'),
            ('haridwar_road',      'sector_17',          'Ring Rd South',    'city'),
            ('sector_17',          'nehru_colony',       'Nehru Rd',         'city'),
            ('nehru_colony',       'ec_road_mid',        'EC Rd South',      'city'),
            ('race_course',        'prem_nagar',         'Haridwar Rd West', 'highway'),
            ('prem_nagar',         'ndrf_camp',          'NH-58',            'highway'),
            ('parade_ground',      'govindgarh',         'Gandhi Rd Ext',    'residential'),
        ]

        for u, v, road_name, road_type in roads:
            self._add_edge(G, u, v, road_name, road_type)

        return G

    def _add_edge(self, G: nx.Graph, u: str, v: str, road_name: str, road_type: str):
        """Auto-compute distance from haversine; set all cost attributes."""
        lat1, lon1 = self.nodes[u]
        lat2, lon2 = self.nodes[v]
        dist  = _haversine(lat1, lon1, lat2, lon2)           # km (true)
        speed = self._SPEED.get(road_type, 30)
        t_min = (dist / speed) * 60.0                         # travel minutes

        G.add_edge(u, v,
            distance_km=round(dist, 3),
            time_minutes=round(t_min, 2),
            road_name=road_name,
            road_type=road_type,
            blocked=False,
            flood_risk=0.0,       # updated via update_road_risks()
            road_damage=0.0,      # updated via update_road_risks()
            traffic_factor=1.0,   # 1.0 = free flow
        )

    # ─────────────────────────────────────────────────────────────────
    # GIS integration
    # ─────────────────────────────────────────────────────────────────

    def update_road_risks(self, risk_map: dict):
        """
        Receive flood/damage data from GIS Agent and update edge attributes.

        Parameters
        ----------
        risk_map : dict
            { road_name: {'flood_risk': 0.0–1.0, 'road_damage': 0.0–1.0} }

        Example
        -------
            agent.update_road_risks({
                'Rajpur Rd':  {'flood_risk': 0.80, 'road_damage': 0.40},
                'Station Rd': {'flood_risk': 0.50, 'road_damage': 0.20},
            })
        """
        now = time.time()
        for u, v, d in self.graph.edges(data=True):
            road = d.get('road_name', '')
            if road in risk_map:
                risks = risk_map[road]
                self.graph[u][v]['flood_risk']  = float(risks.get('flood_risk',  0.0))
                self.graph[u][v]['road_damage'] = float(risks.get('road_damage', 0.0))
                self._risk_timestamps[road] = now

    def decay_road_risks(self, decay_factor: float = 0.85):
        """
        Reduce all flood/damage values by decay_factor (0–1).
        Call periodically (e.g. every 30 min) so roads recover over time.
        decay_factor=0.85 means each call reduces risk to 85% of prior value.
        """
        decay_factor = max(0.0, min(1.0, decay_factor))
        for u, v, d in self.graph.edges(data=True):
            self.graph[u][v]['flood_risk']  = round(d.get('flood_risk',  0.0) * decay_factor, 3)
            self.graph[u][v]['road_damage'] = round(d.get('road_damage', 0.0) * decay_factor, 3)

    def estimate_incident_road_risks(
        self,
        incident_lat: float,
        incident_lon: float,
        risk_probability: float,
        incident_type: str,
    ) -> dict:
        """
        Heuristic: roads adjacent to the incident node may be affected.
        Returns a risk_map suitable for update_road_risks().
        Water disasters (Flood, Tsunami, Landslide) get higher road risk.
        """
        water_types = {'Flood', 'Tsunami', 'Landslide'}
        if incident_type.title() not in water_types:
            return {}

        incident_node = self.nearest_node(incident_lat, incident_lon)
        risk_map = {}

        for u, v, d in self.graph.edges(incident_node, data=True):
            road = d.get('road_name', '')
            if road:
                risk_map[road] = {
                    'flood_risk':  round(risk_probability * 0.75, 2),
                    'road_damage': round(risk_probability * 0.35, 2),
                }
        return risk_map

    # ─────────────────────────────────────────────────────────────────
    # Road blocking / unblocking
    # ─────────────────────────────────────────────────────────────────

    def block_road(self, node_a: str, node_b: str) -> bool:
        """Mark road as blocked. A* will not route through it."""
        na, nb = node_a.strip().lower(), node_b.strip().lower()
        if self.graph.has_edge(na, nb):
            self.graph[na][nb]['blocked'] = True
            return True
        return False

    def block_road_by_name(self, road_name: str) -> int:
        """Block all edges with the given road_name. Returns count blocked."""
        count = 0
        for u, v, d in self.graph.edges(data=True):
            if d.get('road_name', '') == road_name:
                self.graph[u][v]['blocked'] = True
                count += 1
        return count

    def unblock_road(self, node_a: str, node_b: str) -> bool:
        """Remove block from a road segment."""
        na, nb = node_a.strip().lower(), node_b.strip().lower()
        if self.graph.has_edge(na, nb):
            self.graph[na][nb]['blocked'] = False
            return True
        return False

    def unblock_road_by_name(self, road_name: str) -> int:
        """Unblock all edges with the given road_name. Returns count unblocked."""
        count = 0
        for u, v, d in self.graph.edges(data=True):
            if d.get('road_name', '') == road_name:
                self.graph[u][v]['blocked'] = False
                count += 1
        return count

    def list_blocked_roads(self) -> list:
        """Return list of (node_a, node_b, road_name) for every blocked edge."""
        return [
            (u, v, d['road_name'])
            for u, v, d in self.graph.edges(data=True)
            if d.get('blocked', False)
        ]

    # ─────────────────────────────────────────────────────────────────
    # Routing (A*)
    # ─────────────────────────────────────────────────────────────────

    def find_route(self, start: str, end: str) -> dict:
        """
        A* from start to end.

        Cost function (FIX — now minimises TRAVEL TIME, not distance):
            cost = time_minutes × traffic_factor
                   × (1 + flood_risk×3 + road_damage×2)

        Blocked roads are REMOVED from graph copy — not penalised.

        Returns
        -------
        dict:
            status           – 'OK' | 'NO_PATH' | 'ERROR'
            path             – list of node names
            distance_km      – total distance
            time_minutes     – estimated travel time
            roads_used       – deduplicated road names in route order
            blocked_roads_active – all currently blocked road names
            start, end
        """
        start = start.strip().lower()
        end   = end.strip().lower()

        if start not in self.graph:
            return {'status': 'ERROR', 'message': f"Start node '{start}' not found.",
                    'start': start, 'end': end}
        if end not in self.graph:
            return {'status': 'ERROR', 'message': f"End node '{end}' not found.",
                    'start': start, 'end': end}
        if start == end:
            return {
                'status': 'OK', 'path': [start],
                'distance_km': 0.0, 'time_minutes': 0.0,
                'roads_used': [], 'blocked_roads_active': [],
                'start': start, 'end': end,
            }

        # Build working copy with blocked edges REMOVED (not penalised)
        H = self.graph.copy()
        for u, v in list(H.edges()):
            if H[u][v].get('blocked', False):
                H.remove_edge(u, v)

        # Assign composite weight (time-based for correct ETA optimisation)
        for u, v in H.edges():
            H[u][v]['weight'] = self._composite_cost(H[u][v])

        try:
            path = nx.astar_path(
                H, start, end,
                heuristic=self._heuristic,
                weight='weight',
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {
                'status': 'NO_PATH',
                'message': 'No accessible route — all connecting roads may be blocked.',
                'start': start, 'end': end,
            }

        # Collect realistic stats from ORIGINAL graph attributes
        total_dist = 0.0
        total_time = 0.0
        roads      = []

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            e = self.graph[u][v]
            total_dist += e['distance_km']
            total_time += e['time_minutes'] * e.get('traffic_factor', 1.0)
            roads.append(e['road_name'])

        # Deduplicate road names while preserving order
        seen, roads_dedup = set(), []
        for r in roads:
            if r not in seen:
                seen.add(r)
                roads_dedup.append(r)

        # Collect all currently blocked road names (for dashboard display)
        blocked_names = list({
            d['road_name']
            for _, __, d in self.graph.edges(data=True)
            if d.get('blocked', False)
        })

        return {
            'status':               'OK',
            'path':                 path,
            'distance_km':          round(total_dist, 2),
            'time_minutes':         round(total_time, 1),
            'roads_used':           roads_dedup,
            'blocked_roads_active': blocked_names,
            'start':                start,
            'end':                  end,
        }

    def route_to_incident(
        self,
        resource_node: str,
        incident_lat: float,
        incident_lon: float,
    ) -> dict:
        """Snap incident coordinates to nearest graph node, then route."""
        nearest = self.nearest_node(incident_lat, incident_lon)
        result  = self.find_route(resource_node, nearest)
        result['incident_node'] = nearest
        return result

    def find_best_resource_route(self, resource_nodes: list, end: str) -> dict:
        """
        Try multiple resource nodes; return route with shortest travel time.

        Parameters
        ----------
        resource_nodes : list[str]
            Candidate start nodes (hospitals, rescue centres, etc.)
        end : str
            Incident destination node.
        """
        best_result = None
        best_time   = float('inf')

        for node in resource_nodes:
            node = node.strip().lower()
            if node not in self.graph:
                continue
            result = self.find_route(node, end)
            if result.get('status') == 'OK' and result['time_minutes'] < best_time:
                best_time   = result['time_minutes']
                best_result = result

        return best_result or {
            'status':  'NO_PATH',
            'message': 'No accessible route from any available resource.',
            'end':     end,
        }

    # ─────────────────────────────────────────────────────────────────
    # Cost function & heuristic
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _composite_cost(d: dict) -> float:
        """
        FIX: cost now based on time_minutes (not distance) so A* truly
        minimises travel time.

            cost = time_minutes
                   × traffic_factor
                   × (1 + flood_risk×3 + road_damage×2)

        Examples:
          Normal 0.5 km city road (30 km/h → 1.0 min, no flood):
              cost = 1.0 × 1.0 × 1.0 = 1.0

          Same road, 80% flooded:
              cost = 1.0 × 1.0 × (1 + 2.4) = 3.4   → heavily penalised

          Damaged 50%:
              cost = 1.0 × 1.0 × (1 + 1.0) = 2.0
        """
        t_min   = d.get('time_minutes',   1.0)
        traffic = d.get('traffic_factor', 1.0)
        flood   = d.get('flood_risk',     0.0)
        damage  = d.get('road_damage',    0.0)
        return t_min * traffic * (1.0 + flood * 3.0 + damage * 2.0)

    def _heuristic(self, a: str, b: str) -> float:
        """
        Admissible A* heuristic: straight-line distance / max_speed → minutes.
        Always ≤ true travel time, so A* remains optimal.
        """
        lat1, lon1 = self.nodes.get(a, (0.0, 0.0))
        lat2, lon2 = self.nodes.get(b, (0.0, 0.0))
        dist_km = _haversine(lat1, lon1, lat2, lon2)
        return (dist_km / self._MAX_SPEED) * 60.0   # optimistic minutes

    # ─────────────────────────────────────────────────────────────────
    # Utilities
    # ─────────────────────────────────────────────────────────────────

    def nearest_node(self, lat: float, lon: float) -> str:
        """Haversine-based nearest node (true km, not Euclidean degrees)."""
        best, best_d = None, float('inf')
        for name, (nlat, nlon) in self.nodes.items():
            d = _haversine(lat, lon, nlat, nlon)
            if d < best_d:
                best_d = d
                best   = name
        return best

    def all_nodes(self) -> list:
        return sorted(self.nodes.keys())

    def node_coords(self, name: str):
        return self.nodes.get(name.strip().lower())