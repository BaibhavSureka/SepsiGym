from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

from models import SepsisAction, SepsisObservation, SepsisState
from openenv_compat import OPENENV_AVAILABLE, create_app
from server.sepsis_environment import SepsisTreatmentEnvironment


if OPENENV_AVAILABLE and create_app is not None:
    app = create_app(SepsisTreatmentEnvironment, SepsisAction, SepsisObservation, env_name="sepsi-gym")
else:
    environment = SepsisTreatmentEnvironment()
    app = FastAPI(title="SepsiGym", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metadata")
    def metadata() -> dict:
        return environment.metadata()

    @app.get("/schema")
    def schema() -> dict:
        return {
            "action_schema": SepsisAction.model_json_schema(),
            "observation_schema": SepsisObservation.model_json_schema(),
            "state_schema": SepsisState.model_json_schema(),
        }

    @app.post("/reset")
    def reset(payload: dict | None = None) -> dict:
        task_id = None
        if payload:
            task_id = payload.get("task_id")
        observation = environment.reset(task_id=task_id)
        return {
            "observation": observation.model_dump(),
            "reward": 0.0,
            "done": False,
            "info": {
                "tasks": environment.available_tasks(),
                "metrics": environment.current_metrics(),
            },
        }

    @app.post("/step")
    def step(payload: dict) -> dict:
        action = SepsisAction(**payload)
        observation = environment.step(action)
        return {
            "observation": observation.model_dump(),
            "reward": observation.reward,
            "done": observation.done,
            "info": {
                "metrics": environment.current_metrics(),
            },
        }

    @app.get("/state")
    def state() -> dict:
        return environment.state.model_dump()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>SepsiGym</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 2rem;
                background: #f6f8fb;
                color: #1f2937;
            }
            main {
                max-width: 720px;
                margin: 0 auto;
                background: white;
                border-radius: 16px;
                padding: 2rem;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            }
            h1 {
                margin-top: 0;
            }
            code {
                background: #eef2ff;
                padding: 0.1rem 0.35rem;
                border-radius: 6px;
            }
            ul {
                line-height: 1.7;
            }
            a {
                color: #2563eb;
            }
        </style>
    </head>
    <body>
        <main>
            <h1>SepsiGym</h1>
            <p>This Hugging Face Space is running correctly.</p>
            <p>Available endpoints:</p>
            <ul>
                <li><a href="/health">/health</a></li>
                <li><a href="/metadata">/metadata</a></li>
                <li><a href="/schema">/schema</a></li>
                <li><code>POST /reset</code></li>
                <li><code>POST /step</code></li>
                <li><a href="/state">/state</a></li>
            </ul>
        </main>
    </body>
    </html>
    """


def main() -> None:
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
