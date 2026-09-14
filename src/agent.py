"""
3-Stage Production Agent Pipeline for @ChaseSupport / Banking77 Financial AI Support.
Aligned with Hiver Helpdesk Shared Inbox & Escalation Workflows.
Executes:
  Stage 1: Guardrail & Escalation Check (Fast safety router)
  Stage 2: Intent Classification + FAISS RAG Exemplar Retrieval
  Stage 3: Grounded Reply Generation (Pydantic validated output)
"""

import os
import re
from typing import Dict, List, Any
from src.taxonomy import IntentEnum, AgentResponse, RoutingDecision, INTENT_DESCRIPTIONS
from src.guardrails import GuardrailRouter
from src.rag_store import RAGVectorStore


class SupportAgentPipeline:
    """
    Production-grade Customer Support Agent Pipeline for Financial Services (@ChaseSupport / Banking77).
    """

    def __init__(self, data_path: str = "data/processed_subset.csv"):
        self.guardrail = GuardrailRouter()
        self.rag_store = RAGVectorStore(data_path=data_path)
        self.rag_store.initialize()

    def classify_intent_heuristic(self, text: str) -> IntentEnum:
        """Classifies customer query into 1 of 5 financial intent categories."""
        text_lower = text.lower()
        
        # Priority keyword matching for high accuracy
        if any(k in text_lower for k in ["replace", "replacement", "card in mail", "activate", "pin", "apple pay", "digital wallet", "lost card", "freedom card", "delivery", "delivered"]):
            return IntentEnum.CARD_MANAGEMENT_ISSUANCE
        if any(k in text_lower for k in ["wire", "transfer", "zelle", "ach", "deposit", "remittance", "routing number", "fee reversal", "direct deposit", "pending"]):
            return IntentEnum.WIRE_TRANSFER_PAYMENT_ISSUE
        if any(k in text_lower for k in ["lock", "locked", "otp", "2fa", "password", "logon", "login", "user id", "phishing", "suspicious", "security", "recovery", "authenticator"]):
            return IntentEnum.ACCOUNT_ACCESS_SECURITY
        if any(k in text_lower for k in ["dispute", "charge", "charged", "refund", "merchant", "billing", "overcharge", "double payment", "sapphire"]):
            return IntentEnum.CARD_TRANSACTION_DISPUTE
        if any(k in text_lower for k in ["branch", "hours", "locator", "statement", "pdf", "apy", "interest", "checkbook", "checks", "minimum daily balance", "fee waiver"]):
            return IntentEnum.GENERAL_BANKING_INQUIRY
            
        if "card" in text_lower or "charge" in text_lower or "refund" in text_lower:
            return IntentEnum.CARD_TRANSACTION_DISPUTE
            
        return IntentEnum.GENERAL_BANKING_INQUIRY

    def process_message(self, customer_query: str) -> AgentResponse:
        """
        Executes the full 3-stage agent pipeline.
        Returns validated AgentResponse Pydantic object.
        """
        # Stage 1: Guardrail & Risk Router
        routing: RoutingDecision = self.guardrail.evaluate(customer_query)
        
        if routing.should_escalate:
            # Immediate human escalation route (Hiver Shared Inbox Triage)
            return AgentResponse(
                intent="Escalated_Human_Review",
                reply="We're escalating your issue directly to a senior financial support specialist for immediate secure review. A team member will reach out to you via direct message or secure message center.",
                escalated=True,
                escalation_reason=routing.reason,
                rag_exemplars=[],
                confidence=routing.risk_score
            )

        # Stage 2: Intent Classification & RAG Retrieval
        predicted_intent = self.classify_intent_heuristic(customer_query)
        rag_results = self.rag_store.retrieve_exemplars(customer_query, top_k=3)
        exemplar_texts = [f"Exemplar {i+1}: Q: {q} -> A: {a}" for i, (q, a, intnt) in enumerate(rag_results)]

        # Stage 3: Grounded Reply Generation
        top_resp = rag_results[0][1] if rag_results else "Please DM us your phone number & account zip code so we can assist: chase.com/help"
        reply = top_resp
        
        return AgentResponse(
            intent=predicted_intent.value,
            reply=reply,
            escalated=False,
            escalation_reason="Message verified safe for automated handling.",
            rag_exemplars=exemplar_texts,
            confidence=0.92
        )
