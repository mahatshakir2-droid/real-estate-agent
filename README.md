# Real-estate AI Lead Agent

A LangGraph-based real-estate assistant that handles property searches, lead qualification, FAQ answering (LLM grounded on a static FAQ set), viewing bookings with human-in-the-loop approval, and human handoff. Checkpoint memory keyed by phone number lets returning leads resume their conversation.

## Tech stack

- Python 3.11+ (tested on 3.14.3)
- [LangGraph](https://www.langgraph.dev/) (>= 0.6, uses `interrupt()` for HITL)
- [LangChain](https://python.langchain.com/)
- [LangChain OpenAI](https://python.langchain.com/docs/integrations/providers/openai/)
- [Pydantic v2](https://docs.pydantic.dev/v2/)

## Project structure

```
real-estate-agent/
├── data/
│   ├── listings.json          # 15 fake property listings
│   └── faqs.json              # Static FAQ knowledge base
├── src/lead_agent/
│   ├── __init__.py
│   ├── config.py              # Model selection, helpers (phone masking), paths
│   ├── state.py               # AgentState (TypedDict) + LeadProfile, BookingProposal (Pydantic)
│   ├── tools.py               # search_listings tool (LangChain @tool)
│   ├── prompts.py             # LLM system prompts
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── router.py          # Intent classification (structured output) + routing
│   │   ├── qualifier.py       # Collects budget/location/bedrooms/buy_or_rent
│   │   ├── search.py          # LLM tool-calling → search_listings → formatted results
│   │   ├── viewing.py         # Propose viewing time + interrupt() + approval handling
│   │   ├── faq.py             # LLM answers grounded strictly on faqs.json
│   │   └── handoff.py         # Handoff flag + user notification
│   └── graph.py               # Graph wiring (nodes/edges/conditional routing + MemorySaver)
├── tests/
│   ├── test_step1_tools.py      # Unit test for listings + search_listings
│   ├── test_graph_structure.py  # Key-free structural tests (stubs all LLM calls)
│   └── test_conversations.py    # Live example conversations (requires OPENAI_API_KEY)
├── webhooks/
│   └── whatsapp/README.md       # Webhook placeholder (not implemented; prepare structure only)
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

1. **Create and activate virtual environment**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Configure environment**

Copy `.env.example` to `.env` and fill in your OpenAI key:

```bash
cp .env.example .env  # or copy manually on Windows
```

```env
OPENAI_API_KEY=sk-...
LLM_MODEL=gpt-4o-mini
```

## Run tests

- **Step 1 (tools):** verifies listings load and search filters work correctly.

```bash
.venv\Scripts\python.exe tests\test_step1_tools.py
```

- **Structural tests (no API key needed):** stubs all LLMs to verify routing, qualifier loop (across turns), search tool-calling, memory with `thread_id=phone`, FAQ, handoff, and the viewing `interrupt()`/resume HITL cycle.

```bash
.venv\Scripts\python.exe tests\test_graph_structure.py
```

- **Live conversation examples (requires `OPENAI_API_KEY`):** runs 5 example conversations (progressive qualification, viewing with approval, FAQ, handoff, and returning-lead memory continuity).

```bash
.venv\Scripts\python.exe tests\test_conversations.py
```

## Workflow overview

| Path | Behavior |
|---|---|
| `property_search` | Routed to **qualifier**. Extracts `budget`, `location`, `bedrooms`, `buy_or_rent` across turns. When complete, moves to **search** which calls `search_listings` (tool-calling), returns up to 3 matches, and persists matched `property_id`s in state. |
| `viewing_request` | Routed to **viewing**. Resolves a `property_id` from the message or last search results, proposes a slot, calls `interrupt()` to pause for human approval, then confirms or rejects on resume. Only confirmed bookings are recorded in state. |
| `general_faq` | Routed to **faq**. LLM answers strictly using `data/faqs.json` (grounded). If no answer is covered, it explicitly offers human handoff. |
| `human_handoff` | Routed to **handoff**. Notifies the user and sets `handed_off=True`. Logs the masked phone number (`mask_phone`). |

**Memory:** `MemorySaver` with `thread_id = phone number`. Returning leads using the same phone continue from the previous state. Phone numbers are masked in logs via `config.mask_phone`.

## Design notes

- **Router** uses a Pydantic `RouteDecision` (structured output) with a safe fallback to `general_faq`.
- **Qualifier** doesn't loop within a turn; when fields are missing it ends the turn and waits for the next user message (natural "continue asking" behavior).
- **Search** presents at most 3 matches. Budget matching uses +/-15% tolerance. Tool results include a clear "no matches" message.
- **FAQ** is LLM-grounded on a static JSON.
- **Viewing HITL**: in scripts, the human approval is simulated via `Command(resume={"approved": True/False})`. In production this would come from the webhook/UI.
- **WhatsApp webhook** is only scaffolded (no integration code).

## Assumptions

- **LLM provider:** OpenAI `gpt-4o-mini` via `langchain-openai`, controlled by `LLM_MODEL` env var.
- **Qualifier scope:** only on `property_search`, not on `viewing_request`.
- **LangGraph:** >= 0.6.
- **Checkpoint serialization:** Pydantic types registered with `JsonPlusSerializer` to avoid msgpack warnings.

## Troubleshooting

- **Missing API key:** `test_conversations.py` fails without `OPENAI_API_KEY`. Run `test_step1_tools.py` and `test_graph_structure.py` first (no key needed).
- **Recursion limit:** unlikely with current design; can increase via `config={"configurable": {"recursion_limit": 2000, "thread_id": "..."}}`.
- **Windows paths:** use `.venv\Scripts\python.exe` in PowerShell/CMD.

## Next steps

1. Add `OPENAI_API_KEY` to `.env` and run `tests/test_conversations.py` to see live LLM behavior.
2. Tweak prompts in `src/lead_agent/prompts.py` to match your brand voice.
3. Extend `data/faqs.json` or `listings.json` as needed.
4. Implement the actual WhatsApp webhook in `webhooks/whatsapp/app.py` when ready.

## How it works

Every message goes through a router that picks one of four paths:

1. **property_search** -> the qualifier asks for budget, location, bedrooms and buy/rent until it has all four -> the search node calls the `search_listings` tool
2. **viewing_request** -> proposes a viewing slot -> pauses with `interrupt()` until a human approves -> only then confirms
3. **general_faq** -> answers from `faqs.json`
4. **human_handoff** -> flags the lead and tells them a human agent is taking over

Memory uses LangGraph checkpointing with the phone number as `thread_id`, so a returning lead continues the same conversation. Phone numbers are masked in logs.

## Sample conversation

    User:  Hi, I'm looking for a place.
    Agent: To find the right property, I need a bit more info: what is your budget?
    User:  I want to rent, budget around 90000 per month
    Agent: Which city or neighbourhood are you looking in?
    User:  Kilimani, 2 bedrooms
    Agent: L002 - Kilimani, Nairobi - KSh 85,000 / month - 2 bedrooms - fully furnished with borehole water & backup power.

    User:  Can I view the first one?
    [Paused: human agent approves the proposed slot]
    Agent: Your viewing for L003 is confirmed for Wed 21 Oct 2026 at 11:00 AM.

## Run with a free model (Groq)

Create a `.env` file:

    OPENAI_API_KEY=
    OPENAI_BASE_URL=https://api.groq.com/openai/v1
    LLM_MODEL=openai/gpt-oss-120b

Then run:

    python tests\test_conversations.py

## Roadmap

- WhatsApp Cloud API webhook
- Persistent checkpointer (SQLite/Postgres) so state survives restarts
- Real listings API and calendar integration