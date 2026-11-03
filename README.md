# LangGraph Lab — Building AI Agents with LangGraph

## PhD module: Advanced Artificial Intelligence / Generative AI

Practical lab of the *Advanced Artificial Intelligence / Generative AI* module of the PhD
programme. The notebook builds, step by step, a conversational agent with LangGraph and ends with a
capstone project: an **Intelligent University Assistant** that classifies a student's
question, routes it to a specialised agent, calls tools when needed and remembers the
conversation.

## Contents

| File | Purpose |
|---|---|
| `langgraph_lab.ipynb` | The lab notebook with code **and** the written answers to the conceptual questions |
| `langgraph_lab_Statement.ipynb` | The original, untouched lab statement |
| `src/university_assistant.py` | Stand-alone source code of the capstone assistant (tools, graph, CLI) |
| `diagrams/*.mmd`, `diagrams/*.png` | Mermaid sources and rendered PNGs of every graph built in the lab (exported with `get_graph().draw_mermaid()` / `draw_mermaid_png()`) |
| `REPORT.md` | Short report: answers to the conceptual questions, architecture, design choices, evaluation |
| `requirements.txt` | Python dependencies |

## Setup

Python 3.10 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The model is served through **OpenRouter** (<https://openrouter.ai>), an OpenAI-compatible
gateway with free models, so no paid API is required. Create a free account, generate a key
at <https://openrouter.ai/keys> and export it before starting Jupyter. **Do not write it in
the notebook.**

```bash
export OPENROUTER_API_KEY="sk-or-..."   # Windows PowerShell: $env:OPENROUTER_API_KEY="sk-or-..."
```

If the variable is not set, the notebook prompts for the key with `getpass` (hidden input).

The default model is `nvidia/nemotron-3-ultra-550b-a55b:free`, the largest free model on
OpenRouter that supports tool calling (550B parameters, built for agent orchestration), which
Parts VIII and X need. It is a reasoning model, so answers can take a few seconds longer.
To use another free model, set `OPENROUTER_MODEL`, for example:

```bash
export OPENROUTER_MODEL="thinkingmachines/inkling:free"     # 975B MoE, general reasoning + tools
export OPENROUTER_MODEL="google/gemma-4-31b-it:free"        # lighter, less often saturated
```

Free models are rate-limited and the biggest ones are sometimes saturated
(`503 provider_overloaded`). The notebook passes three models in OpenRouter's `models`
parameter (the maximum allowed), so a failed request is retried automatically on the next
model of the list. If a cell still fails, wait a moment and rerun it.

**Daily quota.** An account without credits gets about 50 free-model requests per day
(error `429 free-models-per-day` once exhausted). A full run of the notebook needs roughly
45 requests, so run it at most once a day, or add a one-time 10 $ credit, which raises the
quota to 1000 free requests per day without the free models consuming it.

## How to run

### The notebook

```bash
source .venv/bin/activate
jupyter notebook langgraph_lab.ipynb     # or: jupyter lab
```

Run the cells from top to bottom (*Kernel → Restart & Run All*). Parts I–II need no API key.
Parts III–X call the free model through OpenRouter; a full run is free but may take a few
minutes because of the rate limits.

To execute it from the command line and keep the outputs in the file:

```bash
jupyter nbconvert --to notebook --execute --inplace langgraph_lab.ipynb
```

### The stand-alone assistant

```bash
source .venv/bin/activate
python src/university_assistant.py              # interactive chat (type "quit" to exit)
python src/university_assistant.py --demo       # runs the 6 demo queries + memory test
python src/university_assistant.py --mermaid    # prints the Mermaid source of the graph
```

### Rendering the diagrams

`show_graph()` in the notebook tries to render a PNG through the Mermaid web service.
Offline, it prints the Mermaid source instead. Any `.mmd` file in `diagrams/` can be pasted
into <https://mermaid.live> and exported as PNG/SVG.

## Design choices

**State.** Every graph uses a `TypedDict` state. The capstone state holds the message list
(with the `add_messages` reducer, so that each node *appends* instead of overwriting) and the
`category` chosen by the classifier.

**Classifier → specialised agents.** A first LLM call labels the question
(`COURSES`, `PROGRAMMING`, `MATHEMATICS`, `ADMINISTRATION`, `GENERAL`). A conditional edge
routes to one of five agent nodes created by a factory (`make_agent`), each with its own
system prompt. Separate prompts give better, more focused answers than one generic prompt
and make it easy to add a category.

**Tools.** Tools are defined with `@tool` and bound to the model with `bind_tools`, so the
model itself decides when a tool is needed. A `ToolNode` executes the calls and the flow
returns to the *same* agent, which writes the final answer. The calculator is implemented
with the `ast` module instead of `eval()`, so only arithmetic can be executed.

**Memory.** The graph is compiled with a `MemorySaver` checkpointer. Each conversation is
identified by a `thread_id`; messages of one thread are persisted and replayed on the next
call, while other threads stay isolated.

**Error handling.** Empty input is caught before any model call and routed to a `fallback`
node. Unknown labels returned by the classifier fall back to `GENERAL`. API exceptions in the
classifier or the agents are caught and turned into a polite message. `ToolNode(...,
handle_tool_errors=True)` converts invalid tool calls into error messages the agent can
recover from instead of crashing the graph.

**Evaluation.** A small labelled test set measures routing accuracy and latency of the
router (Part IX). The report discusses how to make routing more robust (structured output,
few-shot examples, enum validation).

## Deliverables checklist

- [x] `langgraph_lab.ipynb` — notebook executed with outputs, including the written answers
- [x] `README.md` — setup, how to run, design choices
- [x] Source code — `src/university_assistant.py`
- [x] Architecture diagram — `diagrams/assistant.png` / `diagrams/assistant.mmd` (+ the other graphs)
- [ ] Screenshots showing execution — take them after running the notebook with your key
- [x] Short report — `REPORT.md`
