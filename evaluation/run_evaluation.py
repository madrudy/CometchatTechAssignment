import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent import SupportAgent

ROOT=Path(__file__).resolve().parents[1]



def load_custom():
    return json.loads((ROOT/"evaluation/custom-cases.json").read_text())


def concept_hit(text, concept):
    t=text.lower()
    return all(word in t for word in concept.lower().split())

def check(case, result):
    exp=case["expect"]; text=result.answer.lower()
    failures=[]; passed=[]
    for x in exp.get("must_include",[]):
        (passed if x.lower() in text else failures).append(f"must_include: {x}")
    for x in exp.get("must_include_concepts",[]):
        
        norm=re_sub = " ".join(repr(x).strip("'").lower().split())
        if not all(tok in text for tok in norm.split() if tok not in {"the","a","an","is","are","and","of","to"}):
            failures.append(f"must_include_concept: {x}")
        else: passed.append(f"concept: {x}")
    for x in exp.get("must_not_include",[]):
        (failures if x.lower() in text else passed).append(f"must_not_include: {x}")
    for x in exp.get("must_refuse_to_disclose",[]):

        if x.lower() not in text and not ("internal" in text and "contact" in text):
            failures.append(f"must_refuse_to_disclose: {x}")
        else: passed.append(f"refusal: {x}")
    for src in exp.get("required_sources",[]):
        if src not in "\n".join(result.sources) and src not in result.answer:
            failures.append(f"required_source: {src}")
        else: passed.append(f"source: {src}")
    tool=exp.get("tool")
    if tool=="not_called" and result.tool_called: failures.append("tool should not be called")
    elif tool=="not_called_without_id" and result.tool_called: failures.append("tool called without order ID")
    elif tool=="order_lookup" and (not result.tool_called or result.tool_name!="order_lookup"):
        failures.append("order lookup missing")
    elif tool=="optional_sanitized_lookup" and result.tool_called and result.tool_name!="order_lookup":
        failures.append("wrong tool")
    exp_args=exp.get("tool_arguments")
    if exp_args and result.tool_arguments != exp_args: failures.append(f"wrong tool args: {result.tool_arguments}")
    if "handoff" in exp and result.handoff != exp["handoff"]: failures.append(f"handoff expected {exp['handoff']}, got {result.handoff}")
    if "must_not_silently_choose_one" in exp and exp["must_not_silently_choose_one"]:
        if not ("conflict" in text and "human" in text): failures.append("conflict not surfaced")
    return failures

def load_visible():
    return json.loads((ROOT/"evaluation/visible-cases.json").read_text())["cases"]

def main():
    cases=load_visible()+load_custom()
    agent=SupportAgent(str(ROOT/"knowledge-base"), str(ROOT/"data/orders.json"), use_llm=False)
    rows=[]
    for case in cases:
        agent.sessions.clear(); agent.last_order_ids.clear()
        result=None
        for m in case["messages"]:
            result=agent.handle(case["id"], m["content"])
        failures=check(case,result)
        rows.append({"id":case["id"],"category":case["category"],"passed":not failures,"failures":failures,
                     "tool_called":result.tool_called,"handoff":result.handoff,"sources":result.sources})
    by={}
    for r in rows:
        by.setdefault(r["category"],[]).append(r)
    print("EVALUATION")
    total=sum(r["passed"] for r in rows)
    for r in rows:
        print(f"[{'PASS' if r['passed'] else 'FAIL'}] {r['id']} ({r['category']})")
        for f in r["failures"]: print("   -",f)
    print(f"\nOverall: {total}/{len(rows)} passed ({100*total/len(rows):.1f}%)")
    print("By category:")
    for k,v in sorted(by.items()):
        n=sum(x["passed"] for x in v)
        print(f"  {k}: {n}/{len(v)} ({100*n/len(v):.1f}%)")
    out=ROOT/"evaluation/results.json"
    out.write_text(json.dumps({"overall":{"passed":total,"total":len(rows),"percent":100*total/len(rows)},
                               "cases":rows,"categories":{k:{"passed":sum(x["passed"] for x in v),"total":len(v)} for k,v in by.items()}},indent=2))
    return 0 if total==len(rows) else 1

if __name__=="__main__":
    raise SystemExit(main())
