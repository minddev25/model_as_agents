#!/usr/bin/env python3
"""
Enterprise Knowledge Agent Demo

Uses the reusable MultiAgent orchestrator from multiagent.py.
Same functionality as a 400+ line traditional implementation, in ~80 lines.
"""

from multiagent import MultiAgent, Agent, tool

# ═══════════════════════════════════════════════════════════════════════════════
# Mock data (would be real backends in production)
# ═══════════════════════════════════════════════════════════════════════════════

SALES_DATA = [
    {"month": "2025-10", "region": "North", "revenue": 120000},
    {"month": "2025-11", "region": "North", "revenue": 135000},
    {"month": "2025-12", "region": "North", "revenue": 140000},
    {"month": "2025-10", "region": "South", "revenue": 95000},
    {"month": "2025-11", "region": "South", "revenue": 102000},
    {"month": "2025-12", "region": "South", "revenue": 110000},
]

POLICIES = [
    {
        "id": "P001",
        "title": "Remote Work",
        "text": "Up to 3 days/week with manager approval.",
    },
    {
        "id": "P002",
        "title": "Annual Leave",
        "text": "15 days/year. Request 5 days in advance.",
    },
    {
        "id": "P003",
        "title": "Expenses",
        "text": "Under $500: manager approval. Over: finance approval.",
    },
    {
        "id": "P004",
        "title": "Sick Leave",
        "text": "10 days/year. Certificate needed after 2 days.",
    },
]

LEAVE_REQUESTS: list[dict] = []

# ═══════════════════════════════════════════════════════════════════════════════
# Tools — just decorate regular functions
# ═══════════════════════════════════════════════════════════════════════════════


@tool("Query sales database. Use SQL-like filters in the query.")
def query_sales(query: str) -> dict:
    q = query.lower()
    rows = SALES_DATA
    if "north" in q:
        rows = [r for r in rows if r["region"] == "North"]
    if "south" in q:
        rows = [r for r in rows if r["region"] == "South"]
    return {"rows": rows}


@tool("Search company policies by keyword.")
def search_policies(query: str) -> dict:
    q = query.lower()
    # Match any word from query
    words = q.split()
    matches = [
        p for p in POLICIES if any(w in (p["title"] + p["text"]).lower() for w in words)
    ]
    return {"policies": matches or POLICIES}


@tool("Submit a leave request.")
def submit_leave(start_date: str, end_date: str, reason: str) -> dict:
    req_id = f"LR-{1001 + len(LEAVE_REQUESTS)}"
    LEAVE_REQUESTS.append(
        {"id": req_id, "start": start_date, "end": end_date, "reason": reason}
    )
    return {"request_id": req_id, "status": "pending_approval"}


# ═══════════════════════════════════════════════════════════════════════════════
# Agents — just define instructions + tools/handoffs
# ═══════════════════════════════════════════════════════════════════════════════

agents = {
    "supervisor": Agent(
        instructions="You route requests. Transfer to: sales (data/charts), policy (company rules), leave (time off requests). Always transfer, don't answer directly.",
        handoffs=["sales", "policy", "leave"],
    ),
    "sales": Agent(
        instructions="You answer sales questions. Use query_sales to get data, then summarize.",
        tools=[query_sales],
    ),
    "policy": Agent(
        instructions="You answer policy questions. Use search_policies, then cite the policy.",
        tools=[search_policies],
    ),
    "leave": Agent(
        instructions="You handle leave requests. Use submit_leave and confirm the request ID.",
        tools=[submit_leave],
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    app = MultiAgent(agents, supervisor="supervisor")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        result = app.run(query)
        print(f"\n{'─'*60}\nAnswer:\n{result}")
    else:
        # Demo
        for q in [
            "What's our remote work policy?",
            "Show me North region sales",
            "I want to take leave Dec 23-27 for vacation",
        ]:
            result = app.run(q)
            print(f"\n{'─'*60}\nAnswer:\n{result}\n")
