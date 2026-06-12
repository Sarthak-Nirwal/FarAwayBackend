"""
End-to-end pipeline test for all 5 agents.
Run from the disaster_management/ directory:
    python test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agents.gis_agent           import GISAgent
from agents.resource_agent      import ResourceAllocationAgent
from agents.route_agent         import RouteOptimizationAgent
from agents.communication_agent import CommunicationAgent
from agents.feedback_agent      import FeedbackAgent


# ── Sample incident ───────────────────────────────────────────────────

incident = {
    'incident_id':                'INC001',
    'incident_type':              'Flood',
    'severity':                   'High',
    'latitude':                   30.3165,
    'longitude':                  78.0322,
    'location_name':              'Rajpur Road, Dehradun',
    'population_density':         'High',
    'infrastructure_vulnerability':'High',
    'resource_availability':      'Medium',
    'environmental_condition':    'High',
}


def separator(title: str):
    print(f"\n{'═' * 55}")
    print(f"  {title}")
    print('═' * 55)


# ─────────────────────────────────────────────────────────────────────
# 1. GIS Agent
# ─────────────────────────────────────────────────────────────────────
separator("1. GIS AGENT")
gis = GISAgent(resources_path='data/resources.json', maps_dir='maps')
gis_result = gis.analyze(incident)

risk = gis_result['risk_assessment']
print(f"Risk Level    : {risk['risk_level']}")
print(f"Probability   : {risk['probability'] * 100:.1f}%")
print(f"Priority Zone : {gis_result['priority_zone']}")
print(f"Affected Km   : {gis_result['affected_radius_km']} km")
print(f"Nearby Resources ({len(gis_result['nearby_resources'])}):")
for r in gis_result['nearby_resources'][:3]:
    print(f"  {r['name']:<25} {r['distance_km']} km")
print(f"Map saved     : maps/map_INC001.html")


# ─────────────────────────────────────────────────────────────────────
# 2. Resource Allocation Agent
# ─────────────────────────────────────────────────────────────────────
separator("2. RESOURCE ALLOCATION AGENT")
res_agent = ResourceAllocationAgent()
allocation = res_agent.allocate(
    incident_type=incident['incident_type'],
    severity=incident['severity'],
    risk_probability=risk['probability'],
    population_density=incident['population_density'],
    resource_availability=incident['resource_availability'],
    infrastructure_vulnerability=incident['infrastructure_vulnerability'],
    affected_population=8000,
    deduct_from_inventory=True,
)

print(f"Priority Score  : {allocation['priority_score']}/10")
print(f"Priority Level  : {allocation['priority_level']}")
print(f"Required        : {allocation['resources']}")
print(f"Actual deployed : {allocation['actual_resources']}")
print(f"Shortfall       : {allocation['shortfall'] or 'None'}")
print(f"Deployment order: {allocation['deployment_order']}")
print(f"Inventory after : {allocation['inventory_after']}")


# ─────────────────────────────────────────────────────────────────────
# 3. Route Optimization Agent
# ─────────────────────────────────────────────────────────────────────
separator("3. ROUTE OPTIMIZATION AGENT")
route_agent = RouteOptimizationAgent()

# Feed GIS flood risk into road network
road_risks = route_agent.estimate_incident_road_risks(
    incident['latitude'], incident['longitude'],
    risk['probability'], incident['incident_type']
)
route_agent.update_road_risks(road_risks)
print(f"Road risks applied: {list(road_risks.keys())[:3]}")

# Block a road (simulate feedback)
route_agent.block_road_by_name('Station Rd')
print("Blocked: Station Rd")

# Find best route from multiple hospitals
route = route_agent.find_best_resource_route(
    ['doon_hospital', 'coronation_hospital', 'max_hospital'],
    route_agent.nearest_node(incident['latitude'], incident['longitude'])
)
if route.get('status') == 'OK':
    print(f"Best route      : {' → '.join(route['path'])}")
    print(f"Distance        : {route['distance_km']} km")
    print(f"ETA             : {route['time_minutes']} min")
    print(f"Via             : {' → '.join(route['roads_used'])}")
    print(f"Blocked active  : {route['blocked_roads_active']}")
else:
    print(f"Route status    : {route.get('status')} — {route.get('message')}")


# ─────────────────────────────────────────────────────────────────────
# 4. Communication Agent
# ─────────────────────────────────────────────────────────────────────
separator("4. COMMUNICATION AGENT")

# Attach radius to incident for geo-targeting note
incident['affected_radius_km'] = gis_result['affected_radius_km']

comm_agent = CommunicationAgent()   # template mode (no API key)
comms = comm_agent.generate_all(incident, risk, allocation, route)

print("\n--- CITIZEN ALERT ---")
print(comms['citizen_alert'])

print("\n--- SMS ALERT ---")
print(comms['sms_alert'])
print(f"(length: {len(comms['sms_alert'])} chars)")

print("\n--- AUTHORITY REPORT ---")
print(comms['authority_report'])


# ─────────────────────────────────────────────────────────────────────
# 5. Feedback Agent
# ─────────────────────────────────────────────────────────────────────
separator("5. FEEDBACK AGENT")
fb_agent = FeedbackAgent(storage_path='data/feedback.json')

# High-trust report → auto block
e1 = fb_agent.submit('INC001', 'Route Blocked',
                     'Rajpur Rd flooded knee-deep',
                     source='Field Responder', road='Rajpur Rd')
print(f"Report 1 action : {e1['recommended_action']['action']} "
      f"(conf {e1['recommended_action']['confidence']})")

# Citizen report — insufficient alone
e2 = fb_agent.submit('INC001', 'Route Blocked',
                     'Rajpur Rd very bad',
                     source='Citizen', road='Rajpur Rd')
print(f"Report 2 action : {e2['recommended_action']['action']} "
      f"(conf {e2['recommended_action']['confidence']})")

# Duplicate within 10 min — should be caught
e3 = fb_agent.submit('INC001', 'Route Blocked',
                     'Rajpur Rd still flooded',
                     source='Citizen', road='Rajpur Rd')
print(f"Report 3 (dup?) : duplicate={e3.get('duplicate')} — {e3.get('message', '')[:60]}")

# New hazard from different source → corroboration
e4 = fb_agent.submit('INC001', 'New Hazard Spotted',
                     'Landslide near sector 17',
                     source='Police Officer', location='Sector 17')
print(f"Report 4 action : {e4['recommended_action']['action']}")

# Rescue completed
e5 = fb_agent.submit('INC001', 'Rescue Completed',
                     'All civilians evacuated from Zone A',
                     source='NDRF Responder')
print(f"Report 5 action : {e5['recommended_action']['action']}")

# Apply block from feedback to route agent
reroute_needed, triggers = fb_agent.should_reroute('INC001')
if reroute_needed:
    for t in triggers:
        road = t['recommended_action'].get('road')
        if road:
            blocked = route_agent.block_road_by_name(road)
            print(f"Re-optimising: blocked '{road}' ({blocked} edge(s))")

    new_route = route_agent.find_best_resource_route(
        ['doon_hospital', 'coronation_hospital'],
        route_agent.nearest_node(incident['latitude'], incident['longitude'])
    )
    if new_route.get('status') == 'OK':
        print(f"New route ETA   : {new_route['time_minutes']} min "
              f"via {' → '.join(new_route['roads_used'])}")

# Statistics
stats = fb_agent.get_statistics('INC001')
print(f"\nFeedback stats  : {stats}")

separator("ALL TESTS PASSED ✅")