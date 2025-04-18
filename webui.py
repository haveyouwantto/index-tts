import os
import shutil
import sys
import threading
import time

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import gradio as gr
from indextts.utils.webui_utils import next_page, prev_page

from indextts.infer import IndexTTS
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto(language="zh_CN")
MODE = 'local'
tts = IndexTTS(model_dir="checkpoints",cfg_path="checkpoints/config.yaml")

os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

# 获取prompts中的音频文件
def get_prompt_list():
    return [f for f in os.listdir("prompts") if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

# 生成音频
def gen_single(prompt_audio, prompt_choice, text, infer_mode, progress=gr.Progress()):
    output_path = None
    if prompt_audio:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        tts.gr_progress = progress
        if infer_mode == "普通推理":
            output = tts.infer(prompt_audio, text, output_path)  # 普通推理
        else:
            output = tts.infer_fast(prompt_audio, text, output_path)  # 批次推理
    elif prompt_choice:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")
        prompt_audio = os.path.join("prompts", prompt_choice)
        tts.gr_progress = progress
        if infer_mode == "普通推理":
            output = tts.infer(prompt_audio, text, output_path)  # 普通推理
        else:
            output = tts.infer_fast(prompt_audio, text, output_path)  # 批次推理
    return gr.update(value=output, visible=True)

# 更新音频文件列表
def refresh_prompt_list():
    return gr.update(choices=get_prompt_list(), value=get_prompt_list()[0] if get_prompt_list() else None)

with gr.Blocks() as demo:
    mutex = threading.Lock()
    gr.HTML('''<h2><center>IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System</h2>
              <h2><center>(一款工业级可控且高效的零样本文本转语音系统)</h2>
              <p align="center"><a href='https://arxiv.org/abs/2502.05512'><img src='https://img.shields.io/badge/ArXiv-2502.05512-red'></a>''')

    with gr.Tab("音频生成"):
        with gr.Row():
            with gr.Column():
                prompt_audio = gr.Audio(label="请上传参考音频", key="prompt_audio",
                                        sources=["upload", "microphone"], type="filepath")
                prompt_choice = gr.Dropdown(choices=get_prompt_list(), value=None, label="选择音频文件（未上传则使用）", interactive=True)
                refresh_button = gr.Button("刷新列表")

            with gr.Column():
                input_text_single = gr.TextArea(label="请输入目标文本", key="input_text_single")
                infer_mode = gr.Radio(choices=["普通推理", "批次推理"], label="选择推理模式（批次推理：更适合长句，性能翻倍）", value="普通推理")
                gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
            output_audio = gr.Audio(label="生成结果", visible=True, key="output_audio")

        # 刷新按钮功能
        refresh_button.click(fn=refresh_prompt_list, inputs=[], outputs=[prompt_choice])

        # 点击生成按钮
        gen_button.click(fn=gen_single, inputs=[prompt_audio, prompt_choice, input_text_single, infer_mode],
                         outputs=[output_audio])

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="0.0.0.0")
