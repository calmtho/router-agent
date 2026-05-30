import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

import uvicorn

from app.config import config

if __name__ == "__main__":
    print(f"Starting Router Agent on {config.server.host}:{config.server.port}")
    print(f"API available at http://{config.server.host}:{config.server.port}")
    print(f"Health check at http://{config.server.host}:{config.server.port}/health")
    print(f"Chat endpoint at http://{config.server.host}:{config.server.port}/chat")

    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=True,
    )
