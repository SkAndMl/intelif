from datetime import date

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from intelif.errors import IntelifUnsupportedError, IntelifValidationError
from intelif.model import Intelif
from intelif.types import JSONContent, Question

LATEST_ALIAS = "intelif-latest"


class SystemOneRequest(BaseModel):
    model: str
    state: JSONContent
    questions: dict[str, Question] = Field(min_length=1)


def create_app(
    model: Intelif, api_key: str | None = None, release_date: str | None = None
) -> FastAPI:
    app = FastAPI(title="intelif")
    names = {model.name, LATEST_ALIAS}
    released = release_date or date.today().isoformat()  # noqa: DTZ011

    def authorize(authorization: str | None) -> None:
        if api_key and authorization != f"Bearer {api_key}":
            raise HTTPException(status_code=401, detail="invalid API key")

    @app.post("/v1/systemone")
    def system_one(
        request: SystemOneRequest, authorization: str | None = Header(default=None)
    ) -> dict:
        authorize(authorization)

        if request.model not in names:
            raise HTTPException(
                status_code=404, detail=f"unknown model '{request.model}'"
            )

        try:
            response = model.system_one(request.state, request.questions)
        except IntelifUnsupportedError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except IntelifValidationError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

        return response.model_dump(mode="json")

    @app.get("/v1/models")
    def list_models(authorization: str | None = Header(default=None)) -> dict:
        authorize(authorization)
        return {
            "models": [
                {
                    "name": name,
                    "description": f"Intelif decision model ({model.name})",
                    "release_date": released,
                }
                for name in (model.name, LATEST_ALIAS)
            ]
        }

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model.name, "device": model.device}

    return app
