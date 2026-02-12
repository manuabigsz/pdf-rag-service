from fastapi import APIRouter, Security
from uuid import UUID
from typing import List

router = APIRouter(tags=['health'])

@router.get("/")
def health():
    return {"status": "ok"}