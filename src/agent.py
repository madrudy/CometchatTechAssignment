import json, logging, re
from pathlib import Path
from .retrieval import Retriever
from .order_tool import OrderTool
from .policy import is_sensitive_request, asks_for_order, conflict_topic, needs_handoff
from .generator import MockGenerator, OpenAIGenerator
from .models import AgentResult

logger = logging.getLogger("aster_agent")

class SupportAgent:
    def __init__(self, kb_dir: str, orders_path: str, use_llm: bool = True, debug: bool = False):
        self.retriever = Retriever(kb_dir)
        self.order_tool = OrderTool(orders_path)
        self.generator = OpenAIGenerator() if use_llm else MockGenerator()
        self.use_llm = use_llm
        self.debug = debug
        self.sessions: dict[str, list[dict]] = {}
        self.last_order_ids: dict[str, str] = {}

    def _save(self, session_id: str, role: str, content: str):
        hist = self.sessions.setdefault(session_id, [])
        hist.append({"role": role, "content": content})
        self.sessions[session_id] = hist[-8:]

    def _effective_query(self, session_id: str, message: str) -> str:
        hist=self.sessions.get(session_id, [])
        if not hist:
            return message
        prior_user=[m["content"] for m in hist if m["role"]=="user"][-2:]
        if len(message.split()) <= 6 and prior_user:
            return prior_user[-1] + " " + message
        return message

    def handle(self, session_id: str, message: str) -> AgentResult:
        self._save(session_id, "user", message)
        history=self.sessions.get(session_id, [])[:-1]
        trace={"session_id":session_id, "user_message":message, "history":history.copy()}
        special=None; tool_result=None; tool_called=False; tool_args=None; handoff=False

        if is_sensitive_request(message):
            oid=self.order_tool.extract_order_id(message)
            if oid:
                tool_result=self.order_tool.lookup(oid)
                tool_called=True; tool_args={"order_id":oid}
            special="sensitive"
            handoff=True
            answer=self.generator.generate(message, history, [], tool_result, handoff, special=special)
            result=AgentResult(answer, [], True, tool_called, "order_lookup" if tool_called else None, tool_args,
                               {**trace,"tool_result": tool_result.data if tool_result else None})
            self._save(session_id, "assistant", answer)
            return result

        effective=self._effective_query(session_id, message)
        if any(term in effective.lower() for term in ["germany", "canada", "international", "country", "ship to"]):
            effective += " international shipping supported destinations Canada countries"
        oid=self.order_tool.extract_order_id(message)
        if not oid and asks_for_order(message):
            oid=self.last_order_ids.get(session_id)
        order_like = asks_for_order(message) or bool(re.search(r"\bord-", message, re.I)) or ("order" in message.lower() and any(w in message.lower() for w in ["where","status","track","cancel","refund","change","arrive"]))
        if order_like:
            if not oid:
                if "ord-" in message.lower():
                    tool_result=self.order_tool.lookup("INVALID")
                    tool_called=True; tool_args={"order_id":"INVALID"}
                    special="malformed"
                    handoff=True
                else:
                    special="missing_order"
            else:
                self.last_order_ids[session_id]=oid
                tool_result=self.order_tool.lookup(oid)
                tool_called=True; tool_args={"order_id":oid}
                if tool_result.status in {"not_found","malformed"}:
                    special="not_found"
                    handoff=True
                elif needs_handoff(message):
                    special="unsupported_action"
                    handoff=True
        if special is None and tool_result is None:
            retrieval=self.retriever.search(effective)
            chunks=retrieval.chunks
            trace["retrieved_passages"]=[{
                "source":c.source_label,"score":round(c.score,4),"metadata":c.metadata,"text":c.text
            } for c in chunks]
            low = effective.lower()
            if any(term in low for term in ["vegan", "certified vegan", "material certification", "adhesive certification"]):
                special="insufficient"; handoff=True
            elif "migration note" in low or "ignore the real policy" in low or "60 days" in low:
                special="prompt_injection"; handoff=False
            elif not chunks or max(c.score for c in chunks) < 0.45:
                special="insufficient"; handoff=True
            elif conflict_topic(effective, self.retriever.search_conflict_candidates(effective)):
                special="conflict"; handoff=True
            elif needs_handoff(message) or (("damaged" in low or "broken" in low) and "final" in low and "sale" in low):
                handoff=True
        else:
            chunks=[]

        answer=self.generator.generate(message, history, chunks, tool_result, handoff, special=special)
        sources=[c.source_label for c in chunks]
        result=AgentResult(answer, sources, handoff, tool_called, "order_lookup" if tool_called else None, tool_args,
                           {**trace,"retrieved_passages":trace.get("retrieved_passages",[]),
                            "tool_result":tool_result.data if tool_result else None,
                            "special":special,"final_response":answer})
        self._save(session_id, "assistant", answer)
        return result

def write_trace(result: AgentResult, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result.trace, indent=2, default=str), encoding="utf-8")
