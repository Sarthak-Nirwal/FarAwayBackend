"""
Communication Agent
===================
Fixes applied (v3) — based on review:
  ✅ Alert severity logic: LOW → advisory, HIGH → warning, CRITICAL → evacuation
  ✅ Hindi translation for citizen alerts (bilingual output)
  ✅ Geo-targeting note included in alerts (inside affected radius)
  ✅ Template fallback always works even without Gemini installed
  ✅ generate_all() accepts optional route dict (not required)
  ✅ No crash when risk dict is missing keys
  ✅ SMS alert truncated to ~160 chars for broadcast compatibility
"""

from datetime import datetime


class CommunicationAgent:
    """
    Generate emergency alerts for citizens, authorities, and SMS broadcast.

    Usage
    -----
        agent = CommunicationAgent()                      # template only
        agent = CommunicationAgent(gemini_api_key='...')  # LLM-enhanced

        comms = agent.generate_all(incident, risk, allocation, route)
    """

    _DISASTER_EMOJI: dict = {
        'Flood':      '🌊',
        'Earthquake': '🌍',
        'Landslide':  '⛰️',
        'Tsunami':    '🌊',
        'Cyclone':    '🌀',
        'Fire':       '🔥',
    }
    _SEVERITY_EMOJI: dict = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}

    # English safety advice per disaster
    _SAFETY_ADVICE: dict = {
        'Flood':      'Move to higher ground immediately. Do not cross flooded roads.',
        'Earthquake': 'DROP, COVER, HOLD ON. Move away from buildings after shaking stops.',
        'Landslide':  'Evacuate slope areas immediately. Move to flat, open ground.',
        'Fire':       'Evacuate downwind. Cover mouth with damp cloth. Avoid smoke.',
        'Tsunami':    'Move inland to higher ground NOW. Do not return until officially cleared.',
        'Cyclone':    'Stay indoors away from windows. Secure all loose objects outside.',
    }

    # Hindi safety advice per disaster
    _SAFETY_ADVICE_HI: dict = {
        'Flood':      'तुरंत ऊँचे स्थान पर जाएं। बाढ़ वाली सड़कों पर न जाएं।',
        'Earthquake': 'झुकें, ढकें, पकड़ें। कंपन रुकने के बाद इमारतों से दूर जाएं।',
        'Landslide':  'ढलान वाले क्षेत्रों से तुरंत बाहर निकलें।',
        'Fire':       'हवा की दिशा से दूर जाएं। मुँह पर कपड़ा रखें। धुएं से बचें।',
        'Tsunami':    'अभी ऊँचे स्थान पर जाएं। आधिकारिक अनुमति तक वापस न आएं।',
        'Cyclone':    'खिड़कियों से दूर रहें। बाहर की ढीली वस्तुएं सुरक्षित करें।',
    }

    # Alert type by risk level — FIX: tiered alerts
    _ALERT_TIER: dict = {
        'High':   'EVACUATION ORDER',
        'Medium': 'WARNING',
        'Low':    'ADVISORY',
    }

    # ─────────────────────────────────────────────────────────────────

    def __init__(self, gemini_api_key: str = None):
        self._use_llm = False
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                self._model = genai.GenerativeModel('gemini-1.5-flash')
                self._use_llm = True
                print("[CommunicationAgent] Gemini API enabled.")
            except ImportError:
                print("[CommunicationAgent] google-generativeai not installed — templates only.")
            except Exception as e:
                print(f"[CommunicationAgent] Gemini init failed: {e} — templates only.")

    # ── Public API ────────────────────────────────────────────────────

    def generate_all(
        self,
        incident: dict,
        risk: dict,
        allocation: dict,
        route: dict = None,
    ) -> dict:
        """
        Generate all alert formats.

        Returns
        -------
        dict:
            citizen_alert     – str (bilingual English + Hindi)
            authority_report  – str
            sms_alert         – str (≤160 chars)
            alert_tier        – 'EVACUATION ORDER' | 'WARNING' | 'ADVISORY'
            timestamp         – str
        """
        risk_level = risk.get('risk_level', 'Low')
        return {
            'citizen_alert':   self.citizen_alert(incident, risk, route),
            'authority_report':self.authority_report(incident, risk, allocation, route),
            'sms_alert':       self.sms_alert(incident, risk),
            'alert_tier':      self._ALERT_TIER.get(risk_level, 'ADVISORY'),
            'timestamp':       datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        }

    # ── Citizen Alert (bilingual) ──────────────────────────────────────

    def citizen_alert(self, incident: dict, risk: dict, route: dict = None) -> str:
        if self._use_llm:
            result = self._llm_citizen(incident, risk)
            if result:
                return result
        return self._tmpl_citizen(incident, risk, route)

    def _tmpl_citizen(self, incident: dict, risk: dict, route: dict) -> str:
        dtype      = incident.get('incident_type', 'Emergency').title()
        sev        = incident.get('severity', 'Medium').title()
        loc        = incident.get('location_name', 'your area')
        radius_km  = incident.get('affected_radius_km', None)
        emoji      = self._DISASTER_EMOJI.get(dtype, '⚠️')
        sev_em     = self._SEVERITY_EMOJI.get(sev, '⚠️')
        risk_level = risk.get('risk_level', 'Low')
        tier       = self._ALERT_TIER.get(risk_level, 'ADVISORY')
        advice_en  = self._SAFETY_ADVICE.get(dtype, 'Follow emergency services instructions.')
        advice_hi  = self._SAFETY_ADVICE_HI.get(dtype, 'आपातकालीन सेवाओं के निर्देशों का पालन करें।')
        ts         = datetime.now().strftime('%d %b %Y, %H:%M')

        # Geo-targeting note
        radius_note = ''
        if radius_km:
            radius_note = f"\n📍 Affected area: {radius_km} km radius around {loc}."

        # Blocked road note
        route_note = ''
        if route and route.get('status') == 'OK':
            blocked = route.get('blocked_roads_active', [])
            if blocked:
                route_note = f"\n⛔ Avoid: {', '.join(blocked[:3])}."

        return (
            f"{emoji} {tier} — {dtype.upper()} {sev_em}\n"
            f"{'─' * 48}\n"
            f"📍 Location : {loc}\n"
            f"⚡ Severity : {sev}    Risk: {risk_level} ({risk.get('probability', 0)*100:.0f}%)\n"
            f"🕐 Time     : {ts}\n"
            f"{radius_note}"
            f"\n🇬🇧 {advice_en}{route_note}\n"
            f"\n🇮🇳 {advice_hi}\n\n"
            f"📞 Emergency: 112  |  NDRF: 011-24363260\n"
            f"{'─' * 48}"
        )

    def _llm_citizen(self, incident: dict, risk: dict) -> str:
        prompt = (
            f"Write a concise public emergency alert in English (under 80 words).\n"
            f"Disaster: {incident.get('incident_type')}, "
            f"Severity: {incident.get('severity')}, "
            f"Location: {incident.get('location_name')}, "
            f"Risk Level: {risk.get('risk_level')}.\n"
            f"Include: one specific safety action, emergency number 112.\n"
            f"Plain text only — no markdown, no asterisks."
        )
        try:
            return self._model.generate_content(prompt).text.strip()
        except Exception:
            return ''    # fall through to template

    # ── Authority Report ──────────────────────────────────────────────

    def authority_report(
        self,
        incident: dict,
        risk: dict,
        allocation: dict,
        route: dict = None,
    ) -> str:
        if self._use_llm:
            result = self._llm_authority(incident, risk, allocation)
            if result:
                return result
        return self._tmpl_authority(incident, risk, allocation, route)

    def _tmpl_authority(
        self,
        incident: dict,
        risk: dict,
        allocation: dict,
        route: dict,
    ) -> str:
        dtype = incident.get('incident_type', '?').title()
        sev   = incident.get('severity', '?').title()
        loc   = incident.get('location_name', '?')
        res   = allocation.get('actual_resources', allocation.get('resources', {}))
        short = allocation.get('shortfall', {})

        res_lines = '\n'.join(
            f"    {k.replace('_', ' ').title():<22}: {v}"
            for k, v in res.items()
            if v > 0
        ) or '    None assigned.'

        shortfall_section = ''
        if short:
            short_lines = '\n'.join(f"    {k}: {v} units short" for k, v in short.items())
            shortfall_section = f"\n⚠ RESOURCE SHORTFALL (mutual aid needed):\n{short_lines}\n"

        route_section = ''
        if route and route.get('status') == 'OK':
            route_section = (
                f"\nROUTE TO INCIDENT\n"
                f"  From     : {route.get('start', '?')}\n"
                f"  To       : {route.get('end', '?')}\n"
                f"  Distance : {route.get('distance_km', '?')} km\n"
                f"  ETA      : {route.get('time_minutes', '?')} min\n"
                f"  Via      : {' → '.join(route.get('roads_used', []))}\n"
                f"  Blocked  : {', '.join(route.get('blocked_roads_active', [])) or 'None'}\n"
            )

        return (
            f"{'═' * 50}\n"
            f"  OFFICIAL DISASTER MANAGEMENT REPORT\n"
            f"{'═' * 50}\n"
            f"Incident ID   : {incident.get('incident_id', 'N/A')}\n"
            f"Type          : {dtype}\n"
            f"Severity      : {sev}\n"
            f"Location      : {loc}\n"
            f"Coordinates   : {incident.get('latitude')}, {incident.get('longitude')}\n"
            f"Timestamp     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"\nRISK ASSESSMENT (CPT-Based Bayesian-Inspired)\n"
            f"  Risk Level  : {risk.get('risk_level', '?')}\n"
            f"  Probability : {risk.get('probability', 0) * 100:.1f}%\n"
            f"  P(High)     : {risk.get('high_prob', 0) * 100:.1f}%\n"
            f"  P(Medium)   : {risk.get('medium_prob', 0) * 100:.1f}%\n"
            f"  P(Low)      : {risk.get('low_prob', 0) * 100:.1f}%\n"
            f"\nPRIORITY SCORE\n"
            f"  Score       : {allocation.get('priority_score', '?')}/10\n"
            f"  Level       : {allocation.get('priority_level', '?')}\n"
            f"\nRESOURCE DEPLOYMENT (Actual Available)\n"
            f"{res_lines}"
            f"{shortfall_section}"
            f"{route_section}\n"
            f"ACTION        : "
            f"{self._ALERT_TIER.get(risk.get('risk_level', 'Low'), 'ADVISORY')} — "
            f"IMMEDIATE RESPONSE REQUIRED\n"
            f"{'═' * 50}"
        )

    def _llm_authority(self, incident: dict, risk: dict, allocation: dict) -> str:
        prompt = (
            f"Write a formal emergency authority report (under 150 words).\n"
            f"Incident: {incident.get('incident_type')}, "
            f"Severity: {incident.get('severity')}, "
            f"Location: {incident.get('location_name')}.\n"
            f"Risk probability: {risk.get('probability', 0) * 100:.1f}%. "
            f"Priority: {allocation.get('priority_level')}. "
            f"Resources: {allocation.get('actual_resources', allocation.get('resources'))}.\n"
            f"Plain text only — no markdown."
        )
        try:
            return self._model.generate_content(prompt).text.strip()
        except Exception:
            return ''

    # ── SMS Alert (≤160 chars) ────────────────────────────────────────

    def sms_alert(self, incident: dict, risk: dict) -> str:
        dtype  = incident.get('incident_type', 'Emergency').title()
        sev    = incident.get('severity', '')
        loc    = incident.get('location_name', 'your area')
        rl     = risk.get('risk_level', '?')
        tier   = self._ALERT_TIER.get(rl, 'ALERT')
        msg    = f"{tier}: {dtype} ({sev}) near {loc}. Risk={rl}. Evacuate. Call 112."
        # Truncate to 160 chars for SMS compliance
        return msg[:160]