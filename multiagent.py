#!/usr/bin/env python3
"""
Minimal Multi-Agent for GPT-5.2 (No Framework)

This is a reusable orchestrator that implements the "Handoff as Tool" pattern.
To build your own multi-agent system:

1. Define your tools with @tool decorator
2. Define your agents with Agent()
3. Create a MultiAgent and call run()

Example:
    @tool("Search the web")
    def web_search(query: str) -> str: ...

    agents = {
        "supervisor": Agent("Route requests", handoffs=["researcher"]),
        "researcher": Agent("Do research", tools=[web_search]),
    }
    app = MultiAgent(agents, supervisor="supervisor")
    result = app.run("Find info about X")
"""

from __future__ import annotations

import json
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
# Tool decorator — auto-generates OpenAI tool schema from function signature
# ═══════════════════════════════════════════════════════════════════════════════

TYPE_MAP = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def tool(description: str):
    """Decorator to convert a function into an OpenAI tool with auto-generated schema."""

    def decorator(fn: Callable) -> Callable:
        hints = get_type_hints(fn)
        params = inspect.signature(fn).parameters

        properties = {}
        required = []
        for name, param in params.items():
            ptype = hints.get(name, str)
            properties[name] = {"type": TYPE_MAP.get(ptype, "string")}
            if param.default is inspect.Parameter.empty:
                required.append(name)

        fn._tool_schema = {
            "type": "function",
            "name": fn.__name__,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        }
        fn._tool_fn = fn
        return fn

    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# Agent definition
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Agent:
    """
    An agent with instructions, tools, and optional handoffs to other agents.

    Args:
        instructions: System prompt for this agent
        tools: List of @tool decorated functions
        handoffs: List of agent keys this agent can hand off to
    """

    instructions: str
    tools: list[Callable] = field(default_factory=list)
    handoffs: list[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# MultiAgent — the core multi-agent orchestrator
# ═══════════════════════════════════════════════════════════════════════════════


class MultiAgent:
    """
    Multi-agent orchestrator using Responses API.

    Implements "Handoff as Tool" pattern: each agent's handoffs become
    transfer_to_X tools that the model can call to switch agents.
    """

    def __init__(
        self,
        agents: dict[str, Agent],
        supervisor: str = "supervisor",
        model: str = "gpt-5.2",
        verbose: bool = True,
    ):
        self.agents = agents
        self.supervisor = supervisor
        self.model = model
        self.verbose = verbose
        self.client = OpenAI()

        # Build tool registry and handoff map
        self._tool_registry: dict[str, Callable] = {}
        self._handoff_map: dict[str, str] = {}
        self._agent_tools: dict[str, list[dict]] = {}

        for key, agent in agents.items():
            tools_for_agent = []

            # Register work tools
            for t in agent.tools:
                self._tool_registry[t.__name__] = t._tool_fn
                tools_for_agent.append(t._tool_schema)

            # Auto-generate handoff tools
            for target in agent.handoffs:
                target_agent = agents[target]
                tool_name = f"transfer_to_{target}"
                handoff_schema = {
                    "type": "function",
                    "name": tool_name,
                    "description": f"Transfer to {target} agent. {target_agent.instructions[:100]}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {"type": "string", "description": "Why transfer"}
                        },
                        "required": ["reason"],
                        "additionalProperties": False,
                    },
                }
                tools_for_agent.append(handoff_schema)
                self._handoff_map[tool_name] = target

            self._agent_tools[key] = tools_for_agent

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def run(self, query: str, max_turns: int = 10) -> str:
        """Run the multi-agent system on a query. Returns final answer."""

        self._log(f"\n{'='*60}\nUser: {query}\n{'='*60}")

        input_items: list[dict[str, Any]] = [{"role": "user", "content": query}]
        current = self.supervisor

        for _ in range(max_turns):
            agent = self.agents[current]
            tools = self._agent_tools[current]

            self._log(f"\n  [{current}] thinking...")

            resp = self.client.responses.create(
                model=self.model,
                instructions=agent.instructions,
                input=input_items,
                tools=tools if tools else None,
                tool_choice="auto" if tools else "none",
            )

            input_items += resp.output
            calls = [o for o in resp.output if o.type == "function_call"]

            # No tool calls → final answer
            if not calls:
                self._log(f"  [{current}] → final answer")
                return resp.output_text or "(no output)"

            # Process tool calls
            for call in calls:
                name, args = call.name, json.loads(call.arguments or "{}")
                self._log(f"  [{current}] tool: {name}({str(args)[:50]}...)")

                if name in self._handoff_map:
                    # Handoff
                    next_agent = self._handoff_map[name]
                    self._log(f"  [{current}] → handoff to [{next_agent}]")
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps({"ok": True}),
                        }
                    )
                    current = next_agent
                    break
                else:
                    # Execute tool
                    fn = self._tool_registry.get(name)
                    try:
                        result = fn(**args) if fn else {"error": f"Unknown: {name}"}
                    except Exception as e:
                        result = {"error": str(e)}

                    if not isinstance(result, (dict, list)):
                        result = {"result": result}

                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": call.call_id,
                            "output": json.dumps(
                                result, ensure_ascii=False, default=str
                            ),
                        }
                    )

        return "(max turns reached)"
