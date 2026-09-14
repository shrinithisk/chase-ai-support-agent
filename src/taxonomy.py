"""
Taxonomy and Data Validation Schemas for @ChaseSupport / Banking77 Financial AI Support Agent.
Aligned with Hiver Helpdesk Shared Inbox & Ticketing Operations.
Enforces deterministic structured outputs for routing, classification, generation, and judging.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class IntentEnum(str, Enum):
    CARD_TRANSACTION_DISPUTE = "Card_Transaction_Dispute"
    ACCOUNT_ACCESS_SECURITY = "Account_Access_Security"
    WIRE_TRANSFER_PAYMENT_ISSUE = "Wire_Transfer_Payment_Issue"
    CARD_MANAGEMENT_ISSUANCE = "Card_Management_Issuance"
    GENERAL_BANKING_INQUIRY = "General_Banking_Inquiry"


INTENT_DESCRIPTIONS = {
    IntentEnum.CARD_TRANSACTION_DISPUTE: "Unauthorized debit/credit card charges, double billing, merchant refund delays, chargeback requests.",
    IntentEnum.ACCOUNT_ACCESS_SECURITY: "Locked online banking accounts, stolen credentials, OTP 2FA failures, password reset requests.",
    IntentEnum.WIRE_TRANSFER_PAYMENT_ISSUE: "Pending wire transfers, failed ACH deposits, international remittance delays, wire fee inquiries.",
    IntentEnum.CARD_MANAGEMENT_ISSUANCE: "Lost or stolen debit/credit cards, new card activation, replacement card tracking, PIN resets.",
    IntentEnum.GENERAL_BANKING_INQUIRY: "Branch operating hours, interest rates, account fee schedules, monthly statement requests."
}


class RoutingDecision(BaseModel):
    should_escalate: bool = Field(
        description="Whether the customer message should be escalated to a human support representative immediately."
    )
    reason: str = Field(
        description="Detailed explanation for why escalation was triggered or bypassed."
    )
    risk_score: float = Field(
        ge=0.0, le=1.0,
        description="Calculated risk score from 0.0 (safe/auto-handle) to 1.0 (critical risk)."
    )
    trigger_type: str = Field(
        default="none",
        description="Category of trigger: profanity_abuse, legal_threat, financial_fraud, account_security, or none."
    )


class IntentClassification(BaseModel):
    intent: IntentEnum = Field(
        description="Primary intent predicted from the 5 defined financial categories."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score for predicted intent."
    )
    reasoning: str = Field(
        description="Brief justification for why this intent was selected."
    )


class AgentResponse(BaseModel):
    intent: str = Field(
        description="Classified intent tag for incoming customer query."
    )
    reply: str = Field(
        description="Drafted response to customer, grounded in historical banking support policies and resolution templates."
    )
    escalated: bool = Field(
        description="Final routing status (true if escalated to human agent, false if auto-handled)."
    )
    escalation_reason: str = Field(
        description="Reason for escalation or auto-handling approval."
    )
    rag_exemplars: List[str] = Field(
        default_factory=list,
        description="Historical support responses retrieved from FAISS vector store used for grounding."
    )
    confidence: float = Field(
        default=0.90,
        description="Overall pipeline confidence score."
    )


class JudgeEvaluation(BaseModel):
    faithfulness_score: int = Field(
        ge=1, le=5,
        description="1-5 rating on whether reply adheres strictly to bank compliance without hallucination."
    )
    tone_score: int = Field(
        ge=1, le=5,
        description="1-5 rating on alignment with professional, empathetic, secure financial support voice."
    )
    actionability_score: int = Field(
        ge=1, le=5,
        description="1-5 rating on whether the reply provides actionable next steps (secure portal link, phone verification)."
    )
    overall_score: float = Field(
        ge=1.0, le=5.0,
        description="Unweighted average of quality dimension scores."
    )
    reasoning: str = Field(
        description="Detailed qualitative feedback from LLM Judge."
    )
