"""Quick test for PaddleOCR API - uses a pre-generated mock PDF from fixtures/"""
import httpx
import json
import time
import sys
import os
from pathlib import Path

from app.utils.file_processor import FileProcessor

# --- Read the pre-generated mock PDF from fixtures ---
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
MOCK_PDF_PATH = FIXTURES_DIR / "rag_test_document.pdf"

if not MOCK_PDF_PATH.exists():
    print(f"ERROR: Mock PDF not found at {MOCK_PDF_PATH}")
    sys.exit(1)

pdf_bytes = MOCK_PDF_PATH.read_bytes()

# Read token from environment variable
token = os.environ.get("PADDLEOCR_ACCESS_TOKEN", "")
if not token:
    # Fallback: try loading from .env file
    env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("PADDLEOCR_ACCESS_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
    if not token:
        print("ERROR: PADDLEOCR_ACCESS_TOKEN not set in environment or .env file")
        sys.exit(1)

with open("_paddle_test_log.txt", "w", encoding="utf-8") as log:
    log.write(f"PDF size: {len(pdf_bytes)} bytes\n")

    endpoint = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    headers = {"Authorization": f"Bearer {token}"}
    files = {"file": ("test.pdf", pdf_bytes, "application/pdf")}
    data = {"model": "PaddleOCR-VL-1.6"}

    try:
        with httpx.Client(timeout=120.0) as client:
            log.write("Posting to API...\n")
            resp = client.post(endpoint, headers=headers, files=files, data=data)
            log.write(f"Status: {resp.status_code}\n")
            log.write(f"Response: {resp.text[:1000]}\n")

            job_data = resp.json()
            job_inner = job_data.get("data", job_data)
            job_id = job_inner.get("jobId") or job_inner.get("id") or job_inner.get("job_id")
            log.write(f"jobId: {job_id}\n")

            if not job_id:
                log.write("ERROR: No job ID found!\n")
                sys.exit(1)

            status_url = f"{endpoint}/{job_id}"
            for i in range(60):
                time.sleep(2)
                poll_resp = client.get(status_url, headers=headers)
                poll_data = poll_resp.json()
                data = poll_data.get("data", poll_data)
                state = data.get("state") or data.get("status") or ""
                log.write(f"Poll {i}: state={state}\n")

                if state in ("completed", "success", "done"):
                    result_url_info = data.get("resultUrl", {})
                    json_url = result_url_info.get("jsonUrl") if isinstance(result_url_info, dict) else None

                    markdown_text = ""
                    if json_url:
                        log.write(f"Fetching result from jsonUrl...\n")
                        result_resp = client.get(json_url, timeout=30.0)
                        result_text = result_resp.text
                        log.write(f"Result response length: {len(result_text)} chars\n")
                        try:
                            result = result_resp.json()
                        except Exception:
                            log.write("JSON decode failed, trying JSONL...\n")
                            result = []
                            for line in result_text.strip().splitlines():
                                line = line.strip()
                                if line:
                                    try:
                                        result.append(json.loads(line))
                                    except Exception:
                                        pass

                        markdown_text = FileProcessor._extract_markdown_from_paddle_result(result)
                    else:
                        result_data = data.get("result", data)
                        markdown_text = result_data.get("markdown") or result_data.get("content") or result_data.get("text") or ""
                        if not markdown_text and isinstance(result_data, str):
                            markdown_text = result_data

                    if not markdown_text:
                        log.write(f"Full response: {json.dumps(poll_data, ensure_ascii=False)[:2000]}\n")
                    log.write(f"\nMarkdown length: {len(markdown_text)}\n")
                    log.write(f"Markdown preview:\n{markdown_text[:2000]}\n")
                    break
                elif state in ("failed", "error"):
                    log.write(f"Job failed: {poll_data}\n")
                    break
            else:
                log.write("Timeout after 120s polling\n")

    except Exception as e:
        import traceback
        log.write(f"ERROR: {type(e).__name__}: {e}\n")
        log.write(traceback.format_exc())

print("Done")
