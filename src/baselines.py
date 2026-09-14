"""
Baseline Models for Benchmark Comparison.
Baseline A (Trivial): Majority-class intent prediction ('Card_Transaction_Dispute') + static canned template reply.
Baseline B (Simple): Zero-shot classifier with basic prompt, no safety guardrails, no RAG context.
"""

from src.taxonomy import IntentEnum, AgentResponse


class BaselineATrivial:
    """
    Baseline A (Trivial Model).
    Predicts majority intent ('Card_Transaction_Dispute') for all inputs and outputs a static canned template response.
    Never escalates any message to human support.
    """

    def process_message(self, customer_query: str) -> AgentResponse:
        return AgentResponse(
            intent=IntentEnum.CARD_TRANSACTION_DISPUTE.value,
            reply="Thanks for reaching out to Chase Support! Please log into Online Banking or visit chase.com/help for assistance.",
            escalated=False,
            escalation_reason="Baseline A static policy: Never escalate.",
            rag_exemplars=[],
            confidence=0.50
        )


class BaselineBSimple:
    """
    Baseline B (Simple Model).
    Naive zero-shot pipeline: Predicts intent using basic naive string matching without safety guardrails or RAG exemplars.
    Fails to escalate profanity, legal threats, fraud, or account security incidents.
    """

    def process_message(self, customer_query: str) -> AgentResponse:
        text_lower = customer_query.lower()
        
        # Naive intent prediction
        if "charge" in text_lower or "dispute" in text_lower or "card" in text_lower or "refund" in text_lower:
            intent = IntentEnum.CARD_TRANSACTION_DISPUTE.value
        elif "login" in text_lower or "password" in text_lower or "account" in text_lower:
            intent = IntentEnum.ACCOUNT_ACCESS_SECURITY.value
        elif "wire" in text_lower or "transfer" in text_lower or "zelle" in text_lower:
            intent = IntentEnum.WIRE_TRANSFER_PAYMENT_ISSUE.value
        elif "replace" in text_lower or "pin" in text_lower or "activate" in text_lower:
            intent = IntentEnum.CARD_MANAGEMENT_ISSUANCE.value
        else:
            intent = IntentEnum.GENERAL_BANKING_INQUIRY.value

        return AgentResponse(
            intent=intent,
            reply="We apologize for the inconvenience! Please send us a direct message with your registered phone number and account zip code so we can help.",
            escalated=False,  # Baseline B lacks safety guardrails
            escalation_reason="Baseline B naive policy: Auto-handle everything.",
            rag_exemplars=[],
            confidence=0.70
        )
