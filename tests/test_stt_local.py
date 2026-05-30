"""
test_stt_local.py — 本地 ASR 类库测试（不依赖大模型）
测试 FunASR (Paraformer-zh) 对中文音频的转写效果
"""

import sys
import time
from pathlib import Path

AUDIO_FILE = Path(__file__).parent.parent / "fixtures" / "ai_learn.wav"
if not AUDIO_FILE.exists():
    print(f"音频文件不存在: {AUDIO_FILE}")
    sys.exit(1)


def test_funasr():
    """测试 FunASR (Paraformer) 本地转写"""
    print("\n" + "=" * 60)
    print("Test: FunASR Paraformer (本地)")
    print("=" * 60)
    try:
        from funasr import AutoModel
    except ImportError:
        print("SKIP: 未安装，请运行: pip install funasr modelscope")
        return None

    print("Loading model (首次会下载，约 220MB)...")
    model = AutoModel(
        model="paraformer-zh",  # 中文 Paraformer
        vad_model="fsmn-vad",   # 语音活动检测
        punc_model="ct-punc",   # 标点恢复
    )
    print("Model loaded")

    t0 = time.time()
    result = model.generate(input=str(AUDIO_FILE))
    elapsed = time.time() - t0
    text = result[0]["text"].strip() if result else ""
    print(f"Done in {elapsed:.2f}s")
    print(f"Result:\n{text}")
    return text


def main():
    print("Local ASR Test (FunASR Paraformer-zh)\n")
    print(f"Audio: {AUDIO_FILE.name} ({AUDIO_FILE.stat().st_size/1024:.1f} KB)\n")

    text = test_funasr()

    print("\n" + "=" * 60)
    status = "OK" if text else "SKIP/FAIL"
    preview = (text[:80] + "...") if text and len(text) > 80 else (text or "N/A")
    print(f"Result: [{status}] {preview}")


if __name__ == "__main__":
    main()
