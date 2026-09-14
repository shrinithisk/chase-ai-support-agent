"""
Guardrail & Safety Routing Stage for Financial Customer Support (@ChaseSupport / Banking77 / Hiver Shared Inbox).
Performs fast pre-generation filtering to detect severe risk triggers and route high-risk financial messages to human support representatives.
"""

import re
from typing import Tuple
from src.taxonomy import RoutingDecision


PROFANITY_KEYWORDS = {
    "fuck", "shit", "bitch", "bastard", "trash", "idiot", "idiots",
    "dumb", "hell", "crap", "bullcrap", "ass", "damn", "useless", "worthless", "stealing", "stole", "thief", "thiefs", "scam", "scamming"
}

LEGAL_KEYWORDS = {
    "lawsuit", "lawyer", "attorney", "legal", "cfpb", "consumer financial protection", "attorney general",
    "police report", "cease and desist", "arbitration", "court", "compliance", "police"
}

FINANCIAL_FRAUD_KEYWORDS = {
    "fraud", "fraudulent", "unauthorized charge", "unauthorized wire", "hacked", "drained",
    "stolen identity", "paycheck", "suspicious", "bank notified", "skimmer", "stolen card", "wire fraud",
    "unauthorized debit", "unauthorized ach", "unauthorized", "stolen"
}

ACCOUNT_HIJACK_KEYWORDS = {
    "account hijacked", "hacker changed", "changed my email", "unauthorized access",
    "emergency lock", "credentials leaked", "someone logged into my account", "unauthorized beneficiaries",
    "locked out", "cannot log in", "can't log in", "hacker"
}


class GuardrailRouter:
    """
    Stage 1 Fast Guardrail Router for Financial Support.
    Evaluates incoming customer queries for critical business, security, and legal risks before LLM generation.
    """

    def evaluate(self, customer_query: str) -> RoutingDecision:
        text_lower = customer_query.lower()
        
        # Check Account Hijack / Security Breach
        for kw in ACCOUNT_HIJACK_KEYWORDS:
            if kw in text_lower:
                return RoutingDecision(
                    should_escalate=True,
                    reason=f"Escalation triggered: Account hijack/security breach risk detected ('{kw}').",
                    risk_score=0.98,
                    trigger_type="account_security"
                )

        # Check Legal Threats & CFPB Regulatory Action
        for kw in LEGAL_KEYWORDS:
            if kw in text_lower:
                return RoutingDecision(
                    should_escalate=True,
                    reason=f"Escalation triggered: Formal legal threat or CFPB regulatory complaint ('{kw}').",
                    risk_score=0.95,
                    trigger_type="legal_threat"
                )

        # Check Financial Fraud & Heavy Monetary Loss
        for kw in FINANCIAL_FRAUD_KEYWORDS:
            if kw in text_lower:
                return RoutingDecision(
                    should_escalate=True,
                    reason=f"Escalation triggered: High financial fraud/loss risk detected ('{kw}').",
                    risk_score=0.92,
                    trigger_type="financial_fraud"
                )

        # Check Profanity & Hostile Abuse
        words = re.findall(r'\b\w+\b', text_lower)
        found_profanity = [w for w in words if w in PROFANITY_KEYWORDS]
        if len(found_profanity) >= 1 or "fuck" in text_lower or "shit" in text_lower or "trash" in text_lower:
            return RoutingDecision(
                should_escalate=True,
                reason=f"Escalation triggered: Hostile profanity or abusive language detected.",
                risk_score=0.90,
                trigger_type="profanity_abuse"
            )

        # Safe for automated handling
        return RoutingDecision(
            should_escalate=False,
            reason="Message safe for automated handling.",
            risk_score=0.10,
            trigger_type="none"
        )
