from fastapi import FastAPI


def create_app() -> FastAPI:
    app = FastAPI(
        title="API Validacao de Modelo",
        version="0.1.0",
        description="API base para comparacao e validacao de modelos.",
    )

    @app.get("/", tags=["status"])
    def read_root() -> dict[str, str]:
        return {
            "message": "API de comparacao de modelos online.",
            "docs": "/docs",
        }

    @app.get("/health", tags=["status"])
    def health_check() -> dict[str, str]:
        return {"status": "ok"}

    return app
