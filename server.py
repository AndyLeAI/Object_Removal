"""Start Obj Remover: FastAPI backend + the static UI on http://localhost:5177."""
import logging
import os
import sys
import threading
import webbrowser

PORT = int(os.environ.get("OBJREMOVER_PORT", "5177"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        import uvicorn
        from backend.app import create_app
    except ImportError as err:
        print(f"Missing dependency ({err}). Run `uv sync` first, or start with ./start_mac.sh", file=sys.stderr)
        sys.exit(1)

    url = f"http://localhost:{PORT}"
    print(f"Obj Remover running at {url}")
    print("Keep this window open while using the app. First start downloads the models (~500 MB).")
    if os.environ.get("OBJREMOVER_NO_BROWSER") != "1":
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(), host="127.0.0.1", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
