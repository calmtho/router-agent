"""Vision 端到端测试 - 自动启动服务、测试、清理"""
import json
import os
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

BASE = "http://localhost:8002"
ROOT = Path(__file__).parent


def req(method, path, body=None, file_path=None):
    url = f"{BASE}{path}"
    if method == "GET":
        r = urllib.request.urlopen(url, timeout=30)
        return r.read().decode()

    if method == "POST" and file_path:
        boundary = "----FormBoundary7MA4YWxkTrZu0gW"
        filename = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        data = b""
        data += f"--{boundary}\r\n".encode()
        data += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
        data += b"Content-Type: application/octet-stream\r\n\r\n"
        data += file_bytes + b"\r\n"
        data += f"--{boundary}--\r\n".encode()
        r = urllib.request.Request(url, data=data)
        r.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        resp = urllib.request.urlopen(r, timeout=60)
        return resp.read().decode()

    if method == "POST":
        data = json.dumps(body).encode()
        r = urllib.request.Request(url, data=data)
        r.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(r, timeout=120)
        return resp.read().decode()


def check(name, resp, expected_keys=None):
    print(f"\n{'='*60}")
    print(f"  [{name}]")
    try:
        data = json.loads(resp)
        print(f"  ✓ OK")
        if expected_keys:
            for k in expected_keys:
                val = str(data.get(k, "MISSING!"))[:150]
                print(f"    {k}: {val}")
        return data
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        print(f"    Response: {resp[:300]}")
        return None


# ====== 启动服务器 ======
print("正在启动服务...")
server_proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002", "--log-level", "error"],
    cwd=ROOT,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)

# 等待服务就绪
for i in range(30):
    time.sleep(1)
    try:
        urllib.request.urlopen(f"{BASE}/health", timeout=3)
        print("服务已就绪 ✓")
        break
    except Exception:
        if i == 29:
            print("服务启动超时")
            server_proc.kill()
            sys.exit(1)
        print(f"  等待服务启动... ({i+1}/30)")

# ====== 测试步骤 ======
try:
    # Step 1: 健康检查
    health = req("GET", "/health")
    print(f"\n[Health] {health[:100]}")

    # Step 2: 上传测试图片
    print("\n>> 上传测试图片...")
    resp = req("POST", "/upload/image", file_path=str(ROOT / "fixtures" / "test_image.png"))
    img_data = check("上传图片", resp, ["image_id", "filename"])
    if not img_data:
        raise Exception("图片上传失败")
    image_id = img_data["image_id"]

    # Step 3: 普通对话（验证服务正常）
    print("\n>> 普通对话测试...")
    resp = req("POST", "/chat", body={
        "message": "你好",
        "session_id": "test-vision-demo",
    })
    check("对话测试", resp, ["reply", "agent_used", "cot_reasoning"])

    # Step 4: 图文问答 - 文字识别
    print("\n>> 图文问答测试：图片中的文字...")
    resp = req("POST", "/chat", body={
        "message": "这张图片里写了什么文字？",
        "session_id": "test-vision-demo",
        "image_ids": [image_id],
    })
    r4 = check("图文问答 - 文字识别", resp, ["reply", "agent_used", "cot_reasoning"])
    if r4:
        print(f"\n  >>> 路由: {r4.get('agent_used')}")
        print(f"  >>> 回答: {r4.get('reply', '')[:300]}")

    # Step 5: 图文问答 - 图片描述
    print("\n>> 图文问答测试：图片描述...")
    resp = req("POST", "/chat", body={
        "message": "这张图片里有什么？描述一下画面内容。",
        "session_id": "test-vision-demo",
        "image_ids": [image_id],
    })
    r5 = check("图文问答 - 描述", resp, ["reply", "agent_used"])
    if r5:
        print(f"\n  >>> 路由: {r5.get('agent_used')}")
        print(f"  >>> 回答: {r5.get('reply', '')[:300]}")

    print(f"\n{'='*60}")
    print("  全部测试完成 ✓")

except Exception as e:
    print(f"\n✗ 测试失败: {e}")
finally:
    server_proc.terminate()
    server_proc.wait()
    print("服务已停止")
