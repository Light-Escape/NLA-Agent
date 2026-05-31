"""
Small upload API used by the React frontend.

Run it beside the ADK server when the frontend needs browser-selected files to
be readable by backend tools:

    uvicorn NLA_Master.upload_api:app --port 8001
"""

from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from .upload_store import save_uploaded_matrix_file
except ImportError:
    from upload_store import save_uploaded_matrix_file

app = FastAPI(title="NLA Upload API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/nla/uploads/matrix")
async def upload_matrix_file(file: UploadFile = File(...)) -> dict:
    content = await file.read()
    try:
        metadata = save_uploaded_matrix_file(
            filename=file.filename or "matrix.dat",
            content=content,
            content_type=file.content_type,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"保存上传文件失败: {exc}") from exc

    return {
        "status": "ok",
        "file_id": metadata["file_id"],
        "uri": metadata["uri"],
        "original_name": metadata["original_name"],
        "size": metadata["size"],
        "content_type": metadata.get("content_type", ""),
    }
