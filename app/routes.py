from fastapi import APIRouter, UploadFile, File, Query
from fastapi.responses import JSONResponse
from app.services import process_txt_upload, process_csv_upload

router = APIRouter()

# -------------- upload -----------------
# @router.post("/upload/")
# async def upload_file(file: UploadFile = File(...), include_english: bool = Query(True)):
#     try:
#         results = await process_txt_upload(file, include_english)
#         return {"results": results}
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})

# ------------ upload csv ----------------
@router.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...), include_english: bool = Query(False)):
    try:
        results = await process_csv_upload(file, include_english=include_english)
        return {"results": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})