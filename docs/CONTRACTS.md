# PS-87 — Locked JSON Contracts

**Project:** Smart India Hackathon PS-87 — Digital Mental Health & Psychological Support System for Students

## Status

**LOCKED INTERFACE DOCUMENT**

These objects are the shared interface between Frontend, Backend, ML, and NLP.

### Rules

1. Existing fields must not be removed or renamed without notifying the whole team.
2. Optional fields may be added only with team agreement.
3. Frontend mock JSON must match these shapes.
4. Backend responses must match these shapes.
5. `backend/app/schemas/` must mirror these contracts.
6. If a proposed implementation conflicts with this document, stop and resolve the contract issue before coding.
7. This file defines contract shape; implementation details belong in the relevant technical layer.

---

# 1. User

```json
{
  "id": "string",
  "name": "string",
  "email": "string",
  "created_at": "datetime"
}
```

**Purpose:** Represents the authenticated student/user profile.

---

# 2. AssessmentResult

```json
{
  "user_id": "string",
  "type": "PHQ9 | GAD7 | WHO5",
  "score": "number",
  "severity": "string",
  "submitted_at": "datetime"
}
```

**Purpose:** Stores the result of a PHQ-9, GAD-7, or WHO-5 assessment.

**Related endpoint:** `POST /api/assessments/submit`

**NLP:** Calculate questionnaire score/severity.

**Backend:** Validate, persist, and return the contract-shaped result.

---

# 3. CheckIn

```json
{
  "user_id": "string",
  "type": "mood | sleep | routine",
  "value": "number|string",
  "note": "string|null",
  "timestamp": "datetime"
}
```

**Purpose:** Represents a daily mood, sleep, or routine check-in.

**Related endpoints:**
- `POST /api/checkins/mood`
- `POST /api/checkins/sleep`
- `POST /api/checkins/routine`

---

# 4. DiaryEntry

```json
{
  "user_id": "string",
  "text": "string",
  "sentiment_score": "number",
  "emotion_tags": ["string"],
  "timestamp": "datetime"
}
```

**Purpose:** Represents a student's diary entry and its NLP-derived sentiment/emotion information.

**Related endpoint:** `POST /api/diary/entry`

**NLP:** Sentiment and emotion analysis.

**Backend:** Store the entry and return the contract-shaped result.

---

# 5. RiskScore

```json
{
  "user_id": "string",
  "score": "number 0-100",
  "level": "low | moderate | high",
  "top_factors": [
    {
      "factor": "string",
      "weight": "number"
    }
  ],
  "generated_at": "datetime"
}
```

**Purpose:** Explainable mental-health risk prediction output.

**Related endpoints:**
- `POST /api/risk/score`
- `GET /api/risk/explain/{user_id}`

**ML:** Produce score, level, and contributing factors.

**Backend:** Expose API and persist/return the result.

---

# 6. ChatMessage

```json
{
  "user_id": "string",
  "role": "user | assistant",
  "text": "string",
  "timestamp": "datetime"
}
```

**Purpose:** Represents a user or assistant message in the chat flow.

**Related endpoint:** `POST /api/chat/message`

**Backend/LLM:** Process the user message and return the assistant response.

---

# 7. Recommendation

```json
{
  "user_id": "string",
  "type": "string",
  "text": "string",
  "generated_at": "datetime"
}
```

**Purpose:** Represents an AI-generated recommendation based on the user's current support/risk context.

**Backend/LLM:** Generate and return the recommendation.

---

# 8. Contract Ownership by Layer

| Contract | Main implementation |
|---|---|
| User | Backend |
| AssessmentResult | NLP scoring + Backend persistence |
| CheckIn | Backend |
| DiaryEntry | NLP + Backend |
| RiskScore | ML + Backend |
| ChatMessage | Backend + LLM |
| Recommendation | Backend + LLM |

---

# 9. Layer Boundary

```text
Frontend
    |
    | JSON contracts
    v
Backend API
    |
    +----> NLP
    |
    +----> ML
    |
    +----> LLM
    |
    +----> PostgreSQL
```

Frontend should build against these JSON shapes before the real backend is connected.

NLP and ML should remain independent of FastAPI.

Backend services are adapters/integration glue and must not duplicate NLP or ML business logic.

---

# 10. Current API Surface

## Authentication

- `POST /api/auth/signup`
- `POST /api/auth/login`
- `GET /api/users/me`

## Assessments

- `POST /api/assessments/submit`

## Check-ins

- `POST /api/checkins/mood`
- `POST /api/checkins/sleep`
- `POST /api/checkins/routine`

## Diary

- `POST /api/diary/entry`

## Chat

- `POST /api/chat/message`

## Risk

- `POST /api/risk/score`
- `GET /api/risk/explain/{user_id}`

## Trends

- `GET /api/trends/{id}`

---

# 11. Important Contract Note

The PS-87 project specification explicitly defines the seven locked objects above and gives their core fields. This document preserves those fields.

Where the source material does **not** explicitly define additional request-only fields, response-only fields, validation rules, or database-specific types, do not invent them here.

Those details must be agreed by the team before becoming part of the locked contract.

---

# 12. Change Procedure

If a developer believes a contract must change:

1. Identify the affected object.
2. Explain why the change is required.
3. Check all frontend/backend/ML/NLP consumers.
4. Update this document only after team agreement.
5. Update the corresponding Pydantic schemas.
6. Update frontend mocks.
7. Update API documentation.
8. Test all affected endpoints.

**Never silently change a locked field.**
