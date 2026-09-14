# Golden Evaluation Set Methodology & Sampling Note (Hiver Shared Inbox & Banking77 Support)

## Dataset Overview
The Golden Evaluation Set consists of **200 hand-curated and annotated financial customer support messages** (`@ChaseSupport` / Banking77), specifically aligned with **Hiver's Helpdesk Shared Inbox & Escalation Workflows**. Every example is annotated with ground-truth metadata for Intent Classification, Human Escalation Routing, Risk Trigger Category, and Complexity Tier.

---

## 1. Hiver Alignment & Sampling Strategy
To mirror how Hiver manages email shared inbox ticket triage, the 200 items were sampled using a **stratified complexity matrix**:

| Complexity Tier | Count | Description |
| :--- | :--- | :--- |
| **Standard Financial Queries** | 120 (60%) | Single-intent routine queries distributed evenly across the 5 defined financial intent classes (24 per intent). |
| **Severe Escalation Risk** | 30 (15%) | Critical safety/compliance risk triggers: explicit profanity/abuse, formal legal & CFPB regulatory threats, severe financial fraud/wire loss, and account security breaches. |
| **Multi-Intent Edge Cases** | 25 (12.5%) | Customer inquiries containing overlapping intent categories (e.g. wire delay + account lockout, or card replacement + overdraft fee dispute). |
| **Noisy / Typo Edge Cases** | 25 (12.5%) | Customer noise including severe typos, missing punctuation, all-caps yelling, informal financial slang (`replacmnt`, `opt`, `wer`), and truncated reference IDs. |

---

## 2. Intent Taxonomy & Labeling Rules
Each message was assigned exactly one primary intent label from the locked 5-class financial taxonomy:
1. `Card_Transaction_Dispute`: Unauthorized charges, double billing, merchant refund delays, chargeback requests.
2. `Account_Access_Security`: Locked online banking accounts, stolen credentials, OTP 2FA failures, password reset requests.
3. `Wire_Transfer_Payment_Issue`: Pending wire transfers, failed ACH deposits, international remittance delays, wire fee inquiries.
4. `Card_Management_Issuance`: Lost or stolen debit/credit cards, new card activation, replacement card tracking, PIN resets.
5. `General_Banking_Inquiry`: Branch operating hours, interest rates, account fee schedules, monthly statement requests.

---

## 3. Human Escalation Ground-Truth Guidelines
An incoming customer ticket/message must be labeled `true_escalate = True` if and only if it triggers any of the following 4 critical business risk criteria:
1. **Profanity & Hostile Abuse**: Yelling, severe profanity, or harassment targeting support staff (`profanity_abuse`).
2. **Legal & CFPB Regulatory Threats**: Mentions of lawsuits, CFPB complaints, police reports, or legal counsel (`legal_threat`).
3. **Financial Fraud / High Monetary Loss**: Unauthorized wire transfers > $500, stolen identity, bank fraud alerts, or stolen debit cards (`financial_fraud`).
4. **Account Compromise / Hijack**: Unauthorized email/password change, hacker takeover, or active 2FA bypass attempt (`account_security`).

All routine queries (even dissatisfied customers without profanity or legal threats) are labeled `true_escalate = False` (auto-handle eligible).

---

## 4. Metadata Schema (`evaluation/golden_set_200.csv`)
- `eval_id`: Unique integer identifier (1 to 200).
- `text`: Exact customer query input string.
- `true_intent`: Ground truth intent tag from 5 defined classes.
- `true_escalate`: Boolean (`True` if human escalation required, `False` if auto-handle).
- `trigger_type`: Safety category (`profanity_abuse`, `legal_threat`, `financial_fraud`, `account_security`, or `none`).
- `complexity_tier`: Tier classification (`standard`, `severe_risk`, `multi_intent`, `edge_case`).
