"""Unified runner — single entry point for all AI Broker automation modes.

Usage:
    python -m src.runner                    # Interactive menu
    python -m src.runner auto               # Full autonomous mode
    python -m src.runner scan               # One-time scan
    python -m src.runner scan --tickers AAPL,TSLA
    python -m src.runner execute            # Scan + execute via IBKR
    python -m src.runner monitor            # Watch IBKR positions
    python -m src.runner dashboard          # Launch Streamlit dashboard
    python -m src.runner api                # Start FastAPI server
"""
from __future__ import annotations

import asyncio
import sys


def _print_menu() -> str:
    print(f"""
{'='*55}
  AI Broker — מערכת אוטונומית לניתוח ומסחר מניות
{'='*55}

  בחר מצב הפעלה:

  1) auto       — אוטומציה מלאה (24/7)
  2) auto-adv   — אוטומציה מתקדמת (4 סריקות ביום)
  3) scan       — סריקה חד-פעמית
  4) execute    — סריקה + ביצוע IBKR
  5) monitor    — ניטור פוזיציות IBKR
  6) analyze    — ניתוח מניה בודדת
  7) dashboard  — לוח בקרה (Streamlit)
  8) api        — שרת API (FastAPI)
  9) schedule   — תזמון סריקה יומית
  0) exit       — יציאה

{'='*55}
""")

    choice = input("  בחירה [1-9]: ").strip()
    return choice


def _run_auto(advanced: bool = True):
    from src.automation import AdvancedAutomation, AutonomousEngine
    from src.config.settings import Settings
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("automation.log", encoding="utf-8"),
        ],
    )

    settings = Settings()
    engine = AdvancedAutomation(settings) if advanced else AutonomousEngine(settings)
    asyncio.run(engine.start())


def _run_scan(tickers: str | None = None):
    from src.scanner import run_scan
    from src.config.settings import Settings
    import logging

    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    if tickers:
        settings.watchlist = tickers
    asyncio.run(run_scan(settings))


def _run_execute(tickers: str | None = None, live: bool = False):
    from src.execution import run_execution
    from src.config.settings import Settings
    import logging

    logging.basicConfig(level=logging.INFO)
    settings = Settings()
    if tickers:
        settings.watchlist = tickers
    if live:
        settings.ibkr_port = 7496
        settings.ibkr_trading_enabled = True
    asyncio.run(run_execution(settings))


def _run_monitor():
    from src.monitor import show_status
    import logging

    logging.basicConfig(level=logging.WARNING)
    asyncio.run(show_status())


def _run_analyze(ticker: str):
    from src.cli import analyze
    asyncio.run(analyze(ticker))


def _run_dashboard():
    import subprocess
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def _run_api():
    import subprocess
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"])


def _run_schedule():
    from src.scheduler import run_scheduler
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    asyncio.run(run_scheduler())


def main():
    args = sys.argv[1:]

    if not args:
        choice = _print_menu()
        cmd_map = {
            "1": lambda: _run_auto(advanced=False),
            "2": lambda: _run_auto(advanced=True),
            "3": lambda: _run_scan(),
            "4": lambda: _run_execute(),
            "5": lambda: _run_monitor(),
            "6": lambda: _run_analyze(input("  סימול מניה: ").strip().upper()),
            "7": lambda: _run_dashboard(),
            "8": lambda: _run_api(),
            "9": lambda: _run_schedule(),
            "0": lambda: sys.exit(0),
        }
        handler = cmd_map.get(choice)
        if handler:
            handler()
        else:
            print("בחירה לא תקינה")
        return

    cmd = args[0]

    tickers = None
    live = False
    for i, arg in enumerate(args):
        if arg == "--tickers" and i + 1 < len(args):
            tickers = args[i + 1]
        if arg == "--live":
            live = True

    if cmd == "auto":
        _run_auto(advanced="--advanced" in args or "--adv" in args)
    elif cmd == "auto-adv":
        _run_auto(advanced=True)
    elif cmd == "scan":
        _run_scan(tickers)
    elif cmd == "execute":
        _run_execute(tickers, live)
    elif cmd == "monitor":
        _run_monitor()
    elif cmd == "analyze":
        ticker = args[1] if len(args) > 1 else input("Ticker: ").strip().upper()
        _run_analyze(ticker)
    elif cmd == "dashboard":
        _run_dashboard()
    elif cmd == "api":
        _run_api()
    elif cmd == "schedule":
        _run_schedule()
    else:
        print(f"Unknown command: {cmd}")
        print("Available: auto, auto-adv, scan, execute, monitor, analyze, dashboard, api, schedule")
        sys.exit(1)


if __name__ == "__main__":
    main()
