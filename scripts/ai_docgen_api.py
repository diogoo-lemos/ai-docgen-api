from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, HttpUrl
from typing import Literal, Optional, Dict, Union, List, Tuple
import uuid
import os
import shutil
import time
import requests
import yaml
import json
import openai

app = FastAPI()

# Simulated storage
JOB_STORAGE: Dict[str, Dict] = {}
OUTPUT_DIR = "generated_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# LLM cache (path, method) -> response
LLM_CACHE: Dict[Tuple[str, str, str], str] = {}

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")


# ---------------------------
# Request Models
# ---------------------------
class GenerateRequest(BaseModel):
    schema_url: Optional[HttpUrl] = None
    mode: Literal["docs", "tests", "both"] = "both"
    output_format: Literal["html", "markdown", "zip"] = "zip"
    lang: Optional[str] = "en"
    async_mode: Optional[bool] = False


class UpdateRequest(BaseModel):
    job_id: str
    new_schema_url: HttpUrl
    diff_mode: Optional[bool] = True


# ---------------------------
# Helper Functions
# ---------------------------
def parse_schema_content(content: str, filename: str) -> dict:
    try:
        try:
            return yaml.safe_load(content)
        except yaml.YAMLError:
            return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse uploaded schema: {str(e)}")

def download_and_parse_schema(schema_url: str) -> dict:
    try:
        response = requests.get(schema_url)
        response.raise_for_status()
        content = response.text
        return parse_schema_content(content, schema_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse schema: {str(e)}")

def enrich_with_llm(prompt: str, cache_key: Tuple[str, str, str]) -> str:
    if cache_key in LLM_CACHE:
        return LLM_CACHE[cache_key]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a technical writer that generates detailed API documentation and test case descriptions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        result = response.choices[0].message.content.strip()
        LLM_CACHE[cache_key] = result
        return result
    except Exception as e:
        return f"[LLM Error: {str(e)}]"

def generate_docs_and_tests(job_id: str, request: GenerateRequest, schema: dict):
    time.sleep(2)  # simulate processing time
    job_folder = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(job_folder, exist_ok=True)

    if request.mode in ["docs", "both"]:
        with open(os.path.join(job_folder, "docs.md"), "w") as f:
            f.write("# API Documentation\n\n")
            for path, methods in schema.get("paths", {}).items():
                f.write(f"## {path}\n")
                for method, details in methods.items():
                    summary = details.get("summary", "")
                    desc = details.get("description", "")

                    prompt = f"Provide a developer-friendly explanation for an API endpoint '{method.upper()} {path}' with summary: '{summary}' and description: '{desc}'."
                    cache_key = (path, method, "doc")
                    enriched_text = enrich_with_llm(prompt, cache_key)

                    f.write(f"**{method.upper()}** - {summary}\n\n{desc}\n\n{enriched_text}\n\n")

    if request.mode in ["tests", "both"]:
        with open(os.path.join(job_folder, "test_suite.py"), "w") as f:
            f.write("# Auto-generated test suite\nimport requests\n\n")
            for path, methods in schema.get("paths", {}).items():
                for method, details in methods.items():
                    test_name = f"test_{method}_{path.strip('/').replace('/', '_')}"
                    summary = details.get("summary", "")
                    prompt = f"Write a short and clear comment explaining a test case for an endpoint '{method.upper()} {path}' with summary: '{summary}'."
                    cache_key = (path, method, "test")
                    test_description = enrich_with_llm(prompt, cache_key)

                    f.write(f"def {test_name}():\n")
                    f.write(f"    # {test_description}\n")
                    f.write(f"    # Replace base_url with your API URL\n")
                    f.write(f"    response = requests.{method}('http://localhost:8000{path}')\n")
                    f.write("    assert response.status_code == 200\n\n")

    if request.output_format == "zip":
        shutil.make_archive(job_folder, 'zip', job_folder)

    JOB_STORAGE[job_id]["status"] = "completed"


# ---------------------------
# Routes
# ---------------------------
@app.post("/generate")
def generate(request: GenerateRequest, background_tasks: BackgroundTasks):
    if not request.schema_url:
        raise HTTPException(status_code=400, detail="Missing schema_url or file upload")

    job_id = str(uuid.uuid4())
    JOB_STORAGE[job_id] = {"status": "processing", "request": request.dict()}
    schema = download_and_parse_schema(request.schema_url)

    if request.async_mode:
        background_tasks.add_task(generate_docs_and_tests, job_id, request, schema)
        return {"job_id": job_id, "status": "processing"}
    else:
        generate_docs_and_tests(job_id, request, schema)
        return {
            "job_id": job_id,
            "status": "completed",
            "download_url": f"/result/{job_id}"
        }


@app.post("/upload")
def upload(files: List[UploadFile] = File(...), mode: Literal["docs", "tests", "both"] = "both", output_format: Literal["html", "markdown", "zip"] = "zip"):
    try:
        merged_schema = {"paths": {}, "components": {}}
        errors = []

        for file in files:
            try:
                content = file.file.read().decode("utf-8")
                parsed = parse_schema_content(content, file.filename)
                if "paths" in parsed:
                    merged_schema["paths"].update(parsed["paths"])
                if "components" in parsed:
                    merged_schema.setdefault("components", {}).update(parsed["components"])
            except Exception as e:
                errors.append({"file": file.filename, "error": str(e)})

        if not merged_schema["paths"]:
            raise HTTPException(status_code=400, detail="No valid OpenAPI paths found in uploaded files.")

        job_id = str(uuid.uuid4())
        request = GenerateRequest(schema_url=None, mode=mode, output_format=output_format)
        JOB_STORAGE[job_id] = {"status": "processing", "request": request.dict()}

        generate_docs_and_tests(job_id, request, merged_schema)

        return {
            "job_id": job_id,
            "status": "completed",
            "download_url": f"/result/{job_id}",
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/update")
def update(request: UpdateRequest):
    job = JOB_STORAGE.get(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    diff_summary = "Added 2 endpoints, updated 1 response code"
    job["status"] = "updated"
    job["diff"] = diff_summary
    return {"job_id": request.job_id, "status": "updated", "diff_summary": diff_summary}


@app.get("/status/{job_id}")
def status(job_id: str):
    job = JOB_STORAGE.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, "status": job["status"]}


@app.get("/result/{job_id}")
def result(job_id: str):
    job = JOB_STORAGE.get(job_id)
    if not job or job["status"] != "completed":
        raise HTTPException(status_code=404, detail="Result not ready")

    req = job["request"]
    if req["output_format"] == "zip":
        path = os.path.join(OUTPUT_DIR, f"{job_id}.zip")
        return FileResponse(path, filename=f"{job_id}.zip")
    elif req["output_format"] == "html":
        return JSONResponse(content={"html": "<h1>Generated HTML Docs</h1>"})
    else:
        with open(os.path.join(OUTPUT_DIR, job_id, "docs.md")) as f:
            return JSONResponse(content={"markdown": f.read()})


@app.get("/usage")
def usage():
    return {
        "plan": "Free",
        "docs_generated": 3,
        "tests_generated": 3,
        "limit": 10
    }


@app.get("/plan")
def plan():
    return {
        "plan": "Free",
        "features": {
            "max_requests": 10,
            "formats": ["markdown", "zip"],
            "webhook_support": False
        }
    }
