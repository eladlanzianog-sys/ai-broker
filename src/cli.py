"""CLI entry point for running stock analysis."""
import asyncio
import sys

from src.agents.graph import compile_graph
from src.models.schemas import AnalysisRequest
from src.services.telegram import format_recommendation


async def analyze(ticker: str) -> None:
    print(f"\n{'='*50}")
    print(f"  ניתוח מניה: {ticker.upper()}")
    print(f"{'='*50}\n")

    request = AnalysisRequest(ticker=ticker.upper())
    graph = compile_graph()

    print("⏳ אוסף נתוני שוק...")
    result = await graph.ainvoke({"request": request})

    if result.get("errors"):
        print("\n❌ שגיאות:")
        for err in result["errors"]:
            print(f"  • {err}")
        return

    rec = result.get("recommendation")
    if rec is None:
        print("\n❌ לא התקבלה המלצה")
        return

    print("\n" + format_recommendation(rec))

    print(f"\n{'='*50}")
    print("📋 יומן ביצוע:")
    for entry in result.get("audit_log", []):
        print(f"  {entry}")
    print(f"{'='*50}\n")


def main():
    if len(sys.argv) < 2:
        print("שימוש: python -m src.cli <TICKER>")
        print("דוגמה: python -m src.cli AAPL")
        sys.exit(1)

    ticker = sys.argv[1]
    asyncio.run(analyze(ticker))


if __name__ == "__main__":
    main()
