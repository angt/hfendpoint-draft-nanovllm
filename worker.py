import asyncio
import atexit
import os
import queue
import threading
from datetime import datetime
from typing import Any, Dict

import hfendpoint
import torch.multiprocessing as mp
from huggingface_hub import snapshot_download
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer

def get_int(key: str, default: int) -> int:
    return int(os.environ.get(key, default))

def get_float(key: str, default: float) -> float:
    return float(os.environ.get(key, default))

def get_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() == "true"

CONFIG = {
    "tensor_parallel_size":   get_int("HFENDPOINT_TENSOR_PARALLEL_SIZE", 1),
    "max_model_len":          get_int("HFENDPOINT_MAX_MODEL_LEN", 4096),
    "gpu_memory_utilization": get_float("HFENDPOINT_GPU_MEMORY_UTILIZATION", 0.9),
    "enforce_eager":          get_bool("HFENDPOINT_ENFORCE_EAGER", False),
    "max_num_batched_tokens": get_int("HFENDPOINT_MAX_BATCHED_TOKENS", 32768),
    "max_num_seqs":           get_int("HFENDPOINT_MAX_SEQS", 512),
}

class Worker:
    def __init__(self):
        model_path = snapshot_download(repo_id="Qwen/Qwen3-0.6B")
        self.config = Config(model_path, **CONFIG)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.config.eos = self.tokenizer.eos_token_id
        self.requests = queue.Queue()
        self.notifier = threading.Condition()
        self.loop = None
        self.engine = None
        self.processes = []
        self.events = []
        if self.config.tensor_parallel_size > 1:
            ctx = mp.get_context("spawn")
            for i in range(1, self.config.tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner, args=(self.config, i, event))
                process.start()
                self.processes.append(process)
                self.events.append(event)
        self.model_runner = ModelRunner(self.config, 0, self.events)
        self.scheduler = Scheduler(self.config)
        atexit.register(self.stop)

    def _run(self):
        while True:
            try:
                with self.notifier:
                    self.notifier.wait_for(lambda: not self.requests.empty() or not self.scheduler.is_finished())

                while not self.requests.empty():
                    seq = self.requests.get_nowait()
                    self.scheduler.add(seq)

                sequences, is_prefill = self.scheduler.schedule()
                if not sequences:
                    continue

                new_token_ids = self.model_runner.call("run", sequences, is_prefill)
                self.scheduler.postprocess(sequences, new_token_ids)

                for seq, token_id in zip(sequences, new_token_ids):
                    response_queue = getattr(seq, 'response_queue', None)
                    if not response_queue:
                        continue
                    self.loop.call_soon_threadsafe(response_queue.put_nowait, token_id)
                    if seq.is_finished:
                        self.loop.call_soon_threadsafe(response_queue.put_nowait, None)

            except Exception as e:
                hfendpoint.error(f"worker loop: {e}")

    def submit(self, prompt_token_ids: list[int], sampling_params: SamplingParams) -> asyncio.Queue:
        seq = Sequence(prompt_token_ids, sampling_params)
        seq.response_queue = asyncio.Queue()
        self.requests.put(seq)
        with self.notifier:
            self.notifier.notify()
        return seq.response_queue

    async def start(self):
        self.loop = asyncio.get_running_loop()
        self.engine = threading.Thread(target=self._run, daemon=True)
        self.engine.start()
        await hfendpoint.run()

    def stop(self):
        if hasattr(self, 'model_runner'):
            self.model_runner.call("exit")
        for p in self.processes:
            p.join()

worker = Worker()

@hfendpoint.handler("chat_completions")
async def chat(request_data: Dict[str, Any]):
    prompt_text = worker.tokenizer.apply_chat_template(
        request_data["messages"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )
    prompt_token_ids = worker.tokenizer.encode(prompt_text)

    sampling_params = SamplingParams(
        temperature=request_data.get("temperature", 0.7),
        max_tokens=request_data.get("max_tokens", 2048),
    )
    response = worker.submit(prompt_token_ids, sampling_params)
    decoder = DecodeStream(skip_special_tokens=True)

    while True:
        token_id = await response.get()
        if token_id is None:
            break
        output = decoder.step(worker.tokenizer._tokenizer, token_id)
        if output:
            yield {"content": output}

    yield {"content":"", "finish_reason": "stop"}

if __name__ == "__main__":
    asyncio.run(worker.start())
