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

# Ensure directories exist
os.makedirs("outputs/tasks", exist_ok=True)
os.makedirs("prompts", exist_ok=True)

# Helper: list prompt audio files
def get_prompt_list():
    files = os.listdir("prompts")
    # filter common audio extensions
    return [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg'))]

# Generate speech function
def gen_single(uploaded, choice, text, infer_mode, progress=gr.Progress()):
    # decide which prompt to use
    if uploaded:
        prompt_path = uploaded
    else:
        prompt_path = os.path.join("prompts", choice) if choice else None
    if not prompt_path or not os.path.isfile(prompt_path):
        return gr.update(value=None, visible=False)

    # prepare output path
    timestamp = int(time.time())
    output_path = os.path.join("outputs", f"spk_{timestamp}.wav")

    # set gradio progress
    tts.gr_progress = progress
    # inference
    if infer_mode == "普通推理":
        tts.infer(prompt_path, text, output_path)
    else:
        tts.infer_fast(prompt_path, text, output_path)
    return gr.update(value=output_path, visible=True)

# Update dropdown choices
def refresh_prompts():
    return gr.update(choices=get_prompt_list(), value=get_prompt_list()[0] if get_prompt_list() else None)

# Build UI
gr.Markdown('<h2 align="center">IndexTTS: 工业级可控高效零样本文本转语音系统</h2>')

demo = gr.Blocks()
with demo:
    with gr.Tab("音频生成"):
        with gr.Row():
            prompt_audio = gr.Audio(
                label="上传参考音频",
                sources=["upload", "microphone"],
                type="filepath",
                interactive=True,
            )

            with gr.Column():
                prompt_choice = gr.Dropdown(
                    choices=get_prompt_list(),
                    value=get_prompt_list()[0] if get_prompt_list() else None,
                    label="选择参考音频（若未上传则使用此项）",
                    interactive=True,
                )
                refresh_button = gr.Button("刷新列表")

        input_text_single = gr.TextArea(label="请输入目标文本", key="input_text_single")
        infer_mode = gr.Radio(
            choices=["普通推理", "批次推理"],
            value="普通推理",
            label="选择推理模式（批次推理：更适合长句，性能翻倍）",
        )
        gen_button = gr.Button("生成语音", key="gen_button", interactive=True)
        output_audio = gr.Audio(label="生成结果", visible=False, key="output_audio")

        # Callbacks
        refresh_button.click(
            fn=refresh_prompts,
            inputs=None,
            outputs=[prompt_choice]
        )

        gen_button.click(
            fn=gen_single,
            inputs=[prompt_audio, prompt_choice, input_text_single, infer_mode],
            outputs=[output_audio]
        )

if __name__ == "__main__":
    demo.queue(20)
    demo.launch(server_name="0.0.0.0")
