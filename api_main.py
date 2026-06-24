import os

import uvicorn

from app.api import create_app


app = create_app()


def main() -> None:
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "false").lower() == "true"

    uvicorn.run("api_main:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
