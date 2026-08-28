import argparse, os
from dotenv import load_dotenv
from src.agent import SupportAgent, write_trace

def main():
    load_dotenv()
    p=argparse.ArgumentParser()
    p.add_argument("--session", default="demo")
    p.add_argument("--mock", action="store_true", help="Use deterministic offline response generation.")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--trace", default="logs/trace.json")
    args=p.parse_args()

    root=os.path.dirname(os.path.abspath(__file__))
    agent=SupportAgent(
        os.path.join(root,"knowledge-base"),
        os.path.join(root,"data/orders.json"),
        use_llm=not args.mock,
        debug=args.debug,
    )
    print("Aster & Row Support Agent. Type 'exit' to quit.")
    while True:
        msg=input("\nYou: ").strip()
        if msg.lower() in {"exit","quit"}: break
        result=agent.handle(args.session,msg)
        print(f"\nAgent: {result.answer}")
        if result.sources:
            print("\nSources:")
            for s in result.sources: print(f"  - {s}")
        print(f"\nHuman handoff: {'yes' if result.handoff else 'no'}")
        if args.debug:
            write_trace(result,args.trace)
            print(f"Trace: {args.trace}")

if __name__=="__main__":
    main()
