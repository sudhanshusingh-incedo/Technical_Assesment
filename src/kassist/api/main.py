"""ASGI entrypoint: uvicorn kassist.api.main:app"""

from kassist.api.app import create_app

app = create_app()
