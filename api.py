import gc
import os

import torch

from wdd.AudioProcessor import AudioProcessor

os.environ["HF_HUB_CACHE"] = "./checkpoints/hf_cache"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import shutil
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.middleware.cors import CORSMiddleware  # 引入 CORS中间件模块

from ctc_forced_aligner.alignment_utils import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    load_audio,
)
from ctc_forced_aligner.text_utils import postprocess_results, preprocess_text
from wdd.file_utils import delete_old_files_and_folders, logging
from wdd.model.AlignAudioModel import (
    AlignAudioResponse,
    AlignAudioResult,
    AlignAudioSegment,
    AlignAudioWord,
)
from wdd.TextProcessor import TextProcessor


def get_main_args():
    parser = argparse.ArgumentParser(
        description="CtcForcedAligner WebUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--verbose", action="store_true", default=False, help="Enable verbose mode"
    )
    parser.add_argument(
        "--port", type=int, default=8121, help="Port to run the web UI on"
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to run the web UI on"
    )
    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes"
    )
    # compute related arguments
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="float16" if torch.cuda.is_available() else "float32",
        choices=["bfloat16", "float16", "float32"],
        help="Compute dtype for alignment model inference. Helps with speed and memory usage.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2", None],
        help="Attention implementation for the model. "
        "It will chose the fastest implementation by default.",
    )

    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="if you have a GPU use 'cuda', otherwise 'cpu'",
    )
    return parser.parse_args()  # 每次调用都解析参数


argsMain = get_main_args()

result_dir = "results"

# 设置允许访问的域名
origins = ["*"]  # "*"，即为所有。


TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


# 定义一个函数进行显存清理
def clear_cuda_cache():
    """
    清理PyTorch的显存和系统内存缓存。
    """
    # 强制进行垃圾回收
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            logging.info("Clearing GPU memory...")

            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            # 重置统计信息
            torch.cuda.reset_peak_memory_stats()
            # 打印显存日志
            logging.info(
                f"[GPU Memory] Allocated: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB, "
                f"Max Allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2):.2f} MB"
            )
            logging.info(
                f"[GPU Memory] Reserved: {torch.cuda.memory_reserved() / (1024 ** 2):.2f} MB, "
                f"Max Reserved: {torch.cuda.max_memory_reserved() / (1024 ** 2):.2f} MB"
            )
    except ImportError:
        logging.warning("PyTorch is not installed. Skipping CUDA memory cleanup.")


def align_audio(
    audio_path: str,
    text_path: str,
    language: str,
    romanize: bool = True,
    split_size: str = "word",
    star_frequency: str = "edges",
    merge_threshold: float = 0.00,
    alignment_model: str = "MahmoudAshraf/mms-300m-1130-forced-aligner",
    batch_size: int = 4,
    window_size: int = 30,
    context_size: int = 2,
):
    model, tokenizer = load_alignment_model(
        device=argsMain.device,
        model_path=alignment_model,
        attn_implementation=argsMain.attn_implementation,
        dtype=TORCH_DTYPES[argsMain.compute_dtype],
    )

    audio_waveform = load_audio(audio_path, model.dtype, model.device)

    emissions, stride = generate_emissions(
        model=model,
        audio_waveform=audio_waveform,
        window_length=window_size,
        context_length=context_size,
        batch_size=batch_size,
    )

    with open(text_path, "r") as f:
        lines = f.readlines()
    text = "".join(line for line in lines).replace("\n", " ").strip()

    tokens_starred, text_starred = preprocess_text(
        text, romanize, language, split_size, star_frequency
    )

    segments, scores, blank_token = get_alignments(
        emissions,
        tokens_starred,
        tokenizer,
    )

    spans = get_spans(tokens_starred, segments, blank_token)

    results = postprocess_results(text_starred, spans, stride, scores, merge_threshold)

    # write the results to a file
    save_txt = f"{os.path.splitext(audio_path)[0]}_align.txt"
    with open(save_txt, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"{result['start']}-{result['end']}: {result['text']}\n")

    # write the results to a json file with the whole text and each segment
    save_json = f"{os.path.splitext(audio_path)[0]}_align.json"
    with open(save_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": text,
                "segments": results,
            },
            f,
            ensure_ascii=False,
            indent=4,
        )

    segment_list: list[AlignAudioSegment] = []
    word_list: list[AlignAudioWord] = []

    if results:
        word_list = [
            AlignAudioWord(
                start=word.get("start", 0),
                end=word.get("end", 0),
                word=word.get("text", ""),
            )
            for word in results
        ]

        segment_list.append(
            AlignAudioSegment(
                start=results[0].get("start", 0),
                end=results[-1].get("end", 0),
                text=text,
                words=word_list,
            )
        )

    return AlignAudioResult(
        segments=segment_list,
        full_text=text.strip(),
        duration=results[-1].get("end", 0) if results else 0,
        language=language,
    )


app = FastAPI(
    title="CtcForcedAligner Service",
    version="1.0",
)

# noinspection PyTypeChecker
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # 设置允许的origins来源
    allow_credentials=True,
    allow_methods=["*"],  # 设置允许跨域的http方法，比如 get、post、put等。
    allow_headers=["*"],  # 允许跨域的headers，可以用来鉴别来源等作用。
)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html>
        <head>
            <meta charset=utf-8>
            <title>Api information</title>
        </head>
        <body>
            <a href='./docs'>Documents of API</a>
        </body>
    </html>
    """


@app.get("/test")
async def test():
    """
    测试接口，用于验证服务是否正常运行。
    """
    return PlainTextResponse("success")


@app.post("/process_audio/", response_model=AlignAudioResponse)
async def process_audio(
    audio: UploadFile = File(..., description="上传的音频文件"),
    text: str = Form(..., description="提供的文本提示，必填"),
    language: str = Form(default="", description="语言类型，ISO 639-3代码"),
    romanize: bool = Form(default=True, description="是否进行罗马化"),
    split_size: str = Form(
        default="word", description="分割大小（sentence, word, char）"
    ),
    star_frequency: str = Form(
        default="edges", description="星号频率（edges, segment）"
    ),
    merge_threshold: float = Form(default=0.00, description="合并阈值"),
    alignment_model: str = Form(
        default="MahmoudAshraf/mms-300m-1130-forced-aligner", description="对齐模型"
    ),
    batch_size: int = Form(default=4, description=" 批处理大小"),
    window_size: int = Form(default=30, description=" 窗口大小（秒）"),
    context_size: int = Form(default=2, description=" 上下文大小（秒）"),
):
    """
    处理音频与文本对齐。
    """
    response = AlignAudioResponse()

    # 记录开始时间
    start_time = time.time()

    try:
        logging.info(
            f"text:\n\n{text}\n\n"
            f"language: {language},romanize: {romanize},split_size: {split_size},star_frequency: {star_frequency},merge_threshold: {merge_threshold},"
            f"alignment_model: {alignment_model},batch_size: {batch_size},window_size: {window_size},context_size: {context_size}\n"
        )

        prefix = ""
        # 初始化处理器
        audio_processor = AudioProcessor()
        # 提取上传文件的基础名称（去除扩展名）
        audio_name = Path(audio.filename).stem
        # 构建保存音频文件的目录路径
        audio_dir = os.path.join(result_dir, audio_name)
        # 如果目录已存在，先删除
        if os.path.exists(audio_dir):
            shutil.rmtree(audio_dir)

        os.makedirs(audio_dir, exist_ok=True)  # 如果目录不存在，则创建

        audio_path = await audio_processor.save_upload_to_wav(
            upload_file=audio,
            audio_dir=audio_dir,
            prefix=prefix,
        )
        # 构建保存文件的完整路径
        text_path = os.path.join(audio_dir, f"{prefix}{audio_name}.txt")
        # 如果目标文件已存在，删除旧文件以避免冲突
        if os.path.exists(text_path):
            os.remove(text_path)
        # 将文本写入文件
        with open(text_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        results = align_audio(
            audio_path=audio_path,
            text_path=text_path,
            language=language,
            romanize=romanize,
            split_size=split_size,
            star_frequency=star_frequency,
            merge_threshold=merge_threshold,
            alignment_model=alignment_model,
            batch_size=batch_size,
            window_size=window_size,
            context_size=context_size,
        )
        response.results = results
    except Exception as ex:
        TextProcessor.log_error(ex)
        response.errcode = -1
        response.errmsg = str(ex)
    finally:
        # 删除过期文件
        delete_old_files_and_folders(result_dir, 1)
        clear_cuda_cache()
    # 计算耗时
    elapsed = time.time() - start_time
    logging.info(f"Generation completed in {elapsed}.")

    return response


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    try:
        uvicorn.run(
            app="api:app",
            host=argsMain.host,
            port=argsMain.port,
            workers=argsMain.workers,
            reload=False,
            log_level="info",
        )
    except Exception as e:
        TextProcessor.log_error(e)
        print(e)
        exit(0)
