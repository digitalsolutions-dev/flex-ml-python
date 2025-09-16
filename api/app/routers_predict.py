from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import pandas as pd
import io

from flexml.pipelines import predict_classification, predict_classification_df
from flexml.io_s3 import get_bytes

router = APIRouter(prefix="/predict", tags=["predict"])


# ---------- JSON prediction ----------
class PredictPayload(BaseModel):
    clf_model_key: str
    meta_key: Optional[str] = None
    records: List[Dict[str, Any]]


@router.post("/classification")
def predict_classification_api(body: PredictPayload):
    if not body.records:
        raise HTTPException(status_code=400, detail="records must be a non-empty list")
    try:
        out = predict_classification(
            body.records,
            get_bytes_fn=lambda key: get_bytes(key),
            model_key=body.clf_model_key,
            meta_key=body.meta_key
        )
        return {"ok": True, **out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- CSV prediction ----------
@router.post("/classification/csv")
async def predict_classification_csv(
        clf_model_key: str = Form(...),
        meta_key: Optional[str] = Form(None),
        return_file: bool = Form(False),
        file: UploadFile = File(...)
):
    """
    Accepts a CSV file; returns JSON or downloadable CSV with proba/pred columns appended.
    - form fields: clf_model_key, meta_key (optional), return_file (bool)
    - file field: file
    """
    try:
        content = await file.read()
        # Try pandas read_csv with flexible parsing
        pdf = pd.read_csv(io.BytesIO(content), engine="python")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    try:
        df_out = predict_classification_df(
            pdf,
            get_bytes_fn=lambda key: get_bytes(key),
            clf_model_key=clf_model_key,
            meta_key=meta_key
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if return_file:
        buf = io.BytesIO()
        df_out.to_csv(buf, index=False)
        buf.seek(0)
        filename = (file.filename or "predictions.csv").replace(".csv", "") + "_scored.csv"
        headers = {
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
        return StreamingResponse(buf, media_type="text/csv", headers=headers)
    else:
        # JSON result (heads-up: could be heavy for large files)
        # Return only proba/pred + original row index to keep payload small, if you prefer
        return JSONResponse({
            "ok": True,
            "rows": len(df_out),
            "preview": df_out.head(20).to_dict(orient="records")
        })
