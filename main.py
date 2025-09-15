from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from services.doc_service import process_document
from services.qa_service import answer_query
from services.vector_service import add_to_vectorstore


app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Upload Endpoint ---
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()

        # Process & vectorize
        chunks =  process_document(file_bytes, file.filename)
        add_to_vectorstore(chunks)

        return {
            "status": "success",
            "message": f'Document "{file.filename}" uploaded and processed successfully.',
            "details": {
                "filename": file.filename, 
                "status": "Ready for questions",
                "chunks_created": len(chunks)
            },
        }
    # except ValueError as e:
    #     return {"error": str(e)}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# --- Query Endpoint ---
class QueryRequest(BaseModel):
    question: str
    k: int = 7

@app.post("/query")
async def query_docs(request: QueryRequest):
    try:
        result = await answer_query(request.question, request.k)  # async
        return result
    except Exception as e:
        return {"error": f"Query error: {str(e)}"}

