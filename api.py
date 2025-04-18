import os
import time
import threading
from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from indextts.infer import IndexTTS

# 初始化模型
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")

# 创建 app
app = FastAPI(title="IndexTTS API")

# CORS 支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 确保目录存在
os.makedirs("outputs", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

def get_prompt_path(uploaded_file: UploadFile = None, prompt_name: str = "") -> str:
    if uploaded_file:
        temp_path = f"prompts/temp_{int(time.time())}_{uploaded_file.filename}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.file.read())
        return temp_path
    elif prompt_name:
        path = os.path.join("prompts", prompt_name)
        if os.path.isfile(path):
            return path
    return ""

@app.post("/infer")
async def infer(
    text: str = Form(...),
    prompt_name: str = Form(""),
    uploaded: UploadFile = File(None),
):
    prompt_path = get_prompt_path(uploaded, prompt_name)
    if not prompt_path:
        return JSONResponse({"error": "未提供有效的参考音频"}, status_code=400)

    output_path = f"outputs/spk_{int(time.time())}.wav"
    tts.infer(prompt_path, text, output_path)
    return FileResponse(output_path, media_type="audio/wav")
    
@app.get("/infer")
async def infer_get(
    text: str = Query(..., description="要合成的文本"),
    prompt_name: str = Query(..., description="参考音频文件名（必须在 prompts 文件夹内）")
):
    prompt_path = os.path.join("prompts", prompt_name)
    if not os.path.isfile(prompt_path):
        return JSONResponse({"error": f"找不到参考音频: {prompt_name}"}, status_code=400)

    output_path = f"outputs/spk_{int(time.time())}.wav"
    tts.infer(prompt_path, text, output_path)
    return FileResponse(output_path, media_type="audio/wav")

@app.post("/infer_fast")
async def infer_fast(
    text: str = Form(...),
    prompt_name: str = Form(""),
    uploaded: UploadFile = File(None),
):
    prompt_path = get_prompt_path(uploaded, prompt_name)
    if not prompt_path:
        return JSONResponse({"error": "未提供有效的参考音频"}, status_code=400)

    output_path = f"outputs/spk_{int(time.time())}.wav"
    tts.infer_fast(prompt_path, text, output_path)
    return FileResponse(output_path, media_type="audio/wav")


@app.get("/infer_fast")
async def infer_fast(
    text: str = Query(..., description="要转换为语音的文本"),
    prompt_name: str = Query("", description="参考音频名称"),
):
    """
    使用 GET 请求执行文本到语音的转换。
    不支持上传文件，只能使用预定义的 prompt_name。
    """

    prompt_path = get_prompt_path(prompt_name)
    if not prompt_path:
        return JSONResponse({"error": "未提供有效的参考音频"}, status_code=400)

    output_path = f"outputs/spk_{int(time.time())}.wav"
    tts.infer_fast(prompt_path, text, output_path)  # 使用修改后的tts_infer_fast 函数
    return FileResponse(output_path, media_type="audio/wav", filename="output.wav")  # 返回音频文件


@app.get("/list_prompts")
def list_prompts():
    files = [f for f in os.listdir("prompts") if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]
    return {"prompts": files}

# ✅ 直接运行 Python 文件时自动启动服务
if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=9899)).start()
