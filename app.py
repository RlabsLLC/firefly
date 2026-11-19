import gc
import json
import os
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

ROOT = Path(os.environ.get("STUDIO_ROOT", "/content/llm-studio"))
ROOT.mkdir(parents=True, exist_ok=True)
SESSION = ROOT / ("session-" + uuid.uuid4().hex)
SESSION.mkdir(mode=0o700)
os.environ["HF_HOME"] = str(SESSION / "hf-home")
LLAMA = Path(os.environ.get("LLAMA_CPP", "/content/llama.cpp"))
CONVERTER_PYTHON = os.environ.get("LLAMA_CONVERTER_PYTHON", sys.executable)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
lock = threading.RLock()
stop_event = threading.Event()
state = {
    "phase": "idle", "message": "Ready", "step": 0, "total": 0,
    "loss": None, "eta": None, "logs": [], "run": None, "artifact": None,
}
uploads = {}


def emit(message):
    with lock:
        state["logs"].append(f"{time.strftime('%H:%M:%S')}  {message}")
        state["logs"] = state["logs"][-1000:]


def snapshot():
    with lock:
        return {key: value for key, value in state.items() if key != "run"}


def hardware():
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        major, _ = torch.cuda.get_device_capability(0)
        return {"kind": "cuda", "name": torch.cuda.get_device_name(0),
                "free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2),
                "precision": "bf16" if major >= 8 and torch.cuda.is_bf16_supported() else "fp16",
                "cores": 1}
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr
        if xr.device_type() == "TPU":
            device = xm.xla_device()
            return {"kind": "tpu", "name": str(device), "precision": "bf16",
                    "cores": xr.global_runtime_device_count(), "free_gib": None, "total_gib": None}
    except (ImportError, RuntimeError, OSError) as exc:
        return {"kind": "cpu", "name": "CPU", "precision": "fp32", "cores": 1,
                "free_gib": None, "total_gib": None, "warning": f"TPU unavailable: {exc}"}
    return {"kind": "cpu", "name": "CPU", "precision": "fp32", "cores": 1,
            "free_gib": None, "total_gib": None}


class LoginRequest(BaseModel):
    token: str = Field(min_length=1, max_length=1024)


class Source(BaseModel):
    model: str = Field(min_length=1, max_length=256)
    trust_remote_code: bool = False
    dataset_id: str = Field(default="", max_length=256)
    upload_id: str = Field(default="", max_length=64)
    split: str = Field(default="train", min_length=1, max_length=128)


class Training(Source):
    method: Literal["full", "lora", "qlora", "dora"] = "qlora"
    bits: Literal[4, 8] = 4
    rank: int = Field(default=16, ge=1, le=1024)
    alpha: int = Field(default=32, ge=1, le=4096)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=0.0002, gt=0, le=1)
    batch_size: int = Field(default=1, ge=1, le=128)
    accumulation: int = Field(default=4, ge=1, le=1024)
    max_length: int = Field(default=1024, ge=16, le=131072)
    optimizer: Literal["adamw_torch", "paged_adamw_8bit", "adafactor"] = "adamw_torch"
    epochs: float = Field(default=1, gt=0, le=1000)
    max_steps: int = Field(default=-1, ge=-1)
    gradient_checkpointing: bool = True


class ExportRequest(BaseModel):
    kind: Literal["raw", "merged", "gguf"]
    quant: Literal["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16"] = "Q4_K_M"
    system: str = Field(default="You are a helpful assistant.", max_length=16000)


def load_data(source):
    from datasets import load_dataset
    if bool(source.upload_id) == bool(source.dataset_id.strip()):
        raise ValueError("Choose exactly one uploaded file or Hugging Face dataset ID.")
    if source.upload_id:
        path = uploads.get(source.upload_id)
        if not path or not path.is_file():
            raise ValueError("Upload not found. Upload the dataset again.")
        kind = "parquet" if path.suffix == ".parquet" else "json"
        return load_dataset(kind, data_files=str(path), split="train")
    return load_dataset(source.dataset_id.strip(), split=source.split)


def formatted(row, tokenizer):
    if isinstance(row.get("text"), str) and row["text"].strip():
        text = row["text"]
    elif isinstance(row.get("messages"), list) and row["messages"]:
        if not tokenizer.chat_template:
            raise ValueError("The tokenizer has no chat template; provide a text column instead.")
        text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
    elif isinstance(row.get("prompt"), str) and isinstance(row.get("completion"), str):
        text = row["prompt"] + row["completion"]
    elif isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
        text = "Instruction: " + row["instruction"]
        if row.get("input"):
            text += "\nInput: " + str(row["input"])
        text += "\nResponse: " + row["output"]
    else:
        raise ValueError("Rows must contain text, messages, prompt/completion, or instruction/output.")
    if not text.strip():
        raise ValueError("Dataset contains an empty training example.")
    if tokenizer.eos_token and not text.endswith(tokenizer.eos_token):
        text += tokenizer.eos_token
    return text


def tokenizer_for(source):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(source.model.strip(), trust_remote_code=source.trust_remote_code)


def begin(phase):
    with lock:
        if state["phase"] in ("training", "exporting"):
            raise HTTPException(409, "A job is already running.")
        state.update(phase=phase, message="Starting", step=0, total=0, loss=None, eta=None)


def fail(exc):
    emit("ERROR: " + str(exc))
    emit(traceback.format_exc())
    with lock:
        state.update(phase="error", message=str(exc))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/status")
def status():
    return {"hardware": hardware(), **snapshot()}


@app.get("/api/events")
def events():
    def stream():
        while True:
            yield "data: " + json.dumps(snapshot()) + "\n\n"
            time.sleep(1)
    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/login")
def login_hf(body: LoginRequest):
    from huggingface_hub import HfApi, login
    try:
        user = HfApi().whoami(token=body.token)
        login(token=body.token, add_to_git_credential=False)
        os.environ["HF_TOKEN"] = body.token
    except Exception as exc:
        raise HTTPException(400, "Hugging Face login failed. Check the token and network.") from exc
    return {"username": user["name"]}


@app.get("/api/model")
def model_info(model: str):
    from huggingface_hub import HfApi
    try:
        info = HfApi().model_info(model, expand=["safetensors"])
        data = info.safetensors or {}
        return {"parameters": data.get("total")}
    except Exception as exc:
        raise HTTPException(400, f"Could not inspect model metadata: {exc}") from exc


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".json", ".jsonl", ".parquet"):
        raise HTTPException(400, "Only JSON, JSONL and Parquet files are supported.")
    ident = uuid.uuid4().hex
    path = SESSION / (ident + suffix)
    size = 0
    try:
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 1024**3:
                    raise HTTPException(413, "Upload exceeds the 1 GiB limit.")
                target.write(chunk)
        uploads[ident] = path
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"upload_id": ident, "filename": file.filename}


@app.post("/api/preview")
def preview(source: Source):
    try:
        dataset = load_data(source)
        tokenizer = tokenizer_for(source)
        return {"count": len(dataset), "rows": [formatted(dataset[i], tokenizer) for i in range(min(3, len(dataset)))]}
    except Exception as exc:
        raise HTTPException(400, f"Dataset preview failed: {exc}") from exc


def run_training(config, hw, run_dir):
    try:
        import torch
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainerCallback
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import SFTConfig, SFTTrainer

        emit("Loading dataset and tokenizer...")
        dataset = load_data(config)
        if not len(dataset):
            raise ValueError("Dataset is empty.")
        tokenizer = tokenizer_for(config)
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError("Tokenizer must have a pad or EOS token.")
            tokenizer.pad_token = tokenizer.eos_token
        dataset = dataset.map(lambda row: {"text": formatted(row, tokenizer)},
                              remove_columns=dataset.column_names, desc="Formatting")
        if stop_event.is_set():
            raise RuntimeError("Stopped before training began; no checkpoint was created.")
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[hw["precision"]]
        if config.method == "full" and hw["precision"] == "fp16":
            # AMP's gradient scaler requires trainable FP32 weights on FP16-only GPUs.
            dtype = torch.float32
            emit("Full tuning on FP16 hardware uses FP32 master weights with FP16 mixed precision.")
        kwargs = {"torch_dtype": dtype, "trust_remote_code": config.trust_remote_code}
        if config.method == "qlora":
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=config.bits == 4, load_in_8bit=config.bits == 8,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
            kwargs["device_map"] = {"": 0}
        emit(f"Loading {config.model} ({config.method}, {hw['precision']})...")
        model = AutoModelForCausalLM.from_pretrained(config.model.strip(), **kwargs)
        model.config.use_cache = False
        if config.method != "full":
            if config.method == "qlora":
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.gradient_checkpointing)
            # PEFT's all-linear includes architecture-specific linear projections, excluding the output head.
            linear_count = sum(1 for module in model.modules() if isinstance(module, torch.nn.Linear)
                               or module.__class__.__name__ in ("Linear4bit", "Linear8bitLt"))
            if not linear_count:
                raise ValueError("No compatible linear layers found for adapter injection.")
            emit(f"Auto-targeting all supported linear layers ({linear_count} detected; output head excluded).")
            model = get_peft_model(model, LoraConfig(
                r=config.rank, lora_alpha=config.alpha, lora_dropout=config.dropout,
                target_modules="all-linear", bias="none", task_type="CAUSAL_LM",
                use_dora=config.method == "dora"))
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            emit(f"Trainable parameters: {trainable:,} / {total:,}")
        if stop_event.is_set():
            raise RuntimeError("Stopped before training began; no checkpoint was created.")
        start = time.monotonic()

        class Monitor(TrainerCallback):
            def on_step_end(self, args, trainer_state, control, **kwargs):
                elapsed = time.monotonic() - start
                with lock:
                    state["step"] = trainer_state.global_step
                    state["total"] = trainer_state.max_steps
                    state["eta"] = round(elapsed / trainer_state.global_step *
                                         (trainer_state.max_steps - trainer_state.global_step)) if trainer_state.global_step else None
                if stop_event.is_set():
                    control.should_save = True
                    control.should_training_stop = True
                return control

            def on_log(self, args, trainer_state, control, logs=None, **kwargs):
                if logs and "loss" in logs:
                    with lock:
                        state["loss"] = logs["loss"]
                    emit(f"Step {trainer_state.global_step}/{trainer_state.max_steps}  loss={logs['loss']}")

        args = SFTConfig(
            output_dir=str(run_dir / "checkpoints"), dataset_text_field="text",
            max_length=config.max_length, learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size, gradient_accumulation_steps=config.accumulation,
            num_train_epochs=config.epochs, max_steps=config.max_steps, optim=config.optimizer,
            gradient_checkpointing=config.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if config.gradient_checkpointing else None,
            fp16=hw["precision"] == "fp16", bf16=hw["precision"] == "bf16",
            logging_steps=1, save_strategy="no", report_to="none",
            dataloader_num_workers=0, use_cpu=hw["kind"] == "cpu")
        trainer = SFTTrainer(model=model, args=args, train_dataset=dataset,
                             processing_class=tokenizer, callbacks=[Monitor()])
        emit("Training started. Loss includes all tokens, including prompts.")
        trainer.train()
        raw = run_dir / "raw"
        trainer.save_model(str(raw))
        tokenizer.save_pretrained(str(raw))
        trainer.save_state()
        with lock:
            state["run"] = {"config": config.model_dump(), "raw": str(raw), "dir": str(run_dir)}
            state.update(phase="stopped" if stop_event.is_set() else "complete",
                         message="Stopped and saved" if stop_event.is_set() else "Training complete")
        emit("Saved trained weights and tokenizer. " +
             ("The stop checkpoint also contains optimizer and trainer state." if stop_event.is_set() else ""))
        del trainer, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        fail(exc)


@app.post("/api/train")
def train(config: Training):
    hw = hardware()
    if config.method == "qlora" and hw["kind"] != "cuda":
        raise HTTPException(400, "QLoRA requires a CUDA GPU in this studio.")
    if config.optimizer == "paged_adamw_8bit" and hw["kind"] != "cuda":
        raise HTTPException(400, "Paged 8-bit AdamW requires CUDA.")
    if config.max_steps == 0:
        raise HTTPException(400, "Max steps must be -1 (epochs) or a positive integer.")
    if bool(config.upload_id) == bool(config.dataset_id.strip()):
        raise HTTPException(400, "Choose exactly one dataset source.")
    run_dir = SESSION / ("run-" + uuid.uuid4().hex)
    run_dir.mkdir()
    with lock:
        begin("training")
        stop_event.clear()
        state["run"] = None
        state["artifact"] = None
    threading.Thread(target=run_training, args=(config, hw, run_dir), daemon=True).start()
    return {"ok": True}


@app.post("/api/stop")
def stop():
    with lock:
        if state["phase"] != "training":
            raise HTTPException(409, "No training job is running.")
        stop_event.set()
        state["message"] = "Stopping at the next optimizer step..."
    emit("Stop requested; waiting for the current optimizer step.")
    return {"ok": True}


def run_command(command):
    emit("$ " + " ".join(str(part) for part in command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, bufsize=1, errors="replace")
    assert process.stdout is not None
    for line in process.stdout:
        emit(line.rstrip())
    code = process.wait()
    if code:
        raise RuntimeError(f"Command failed with exit code {code}. See terminal output.")


def merged_model(run):
    raw = Path(run["raw"])
    if run["config"]["method"] == "full":
        return raw
    target = Path(run["dir"]) / "merged"
    if (target / ".complete").exists():
        return target
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    emit("Reloading the unquantized base on CPU and merging adapters; this may require substantial RAM.")
    cfg = run["config"]
    base = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float16,
                                                trust_remote_code=cfg["trust_remote_code"],
                                                low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(base, str(raw)).merge_and_unload()
    target.mkdir(exist_ok=True)
    model.save_pretrained(str(target), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(raw), trust_remote_code=cfg["trust_remote_code"]).save_pretrained(str(target))
    (target / ".complete").touch()
    del model, base
    gc.collect()
    return target


def make_modelfile(model_dir, quant, system):
    config_path = model_dir / "tokenizer_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    chat = config.get("chat_template") or ""
    template_path = model_dir / "chat_template.jinja"
    if not chat and template_path.is_file():
        chat = template_path.read_text(encoding="utf-8")
    if isinstance(chat, list):
        chat = " ".join(item.get("template", "") for item in chat if isinstance(item, dict))
    # Jinja and Ollama Go templates are different languages. Only known templates are translated.
    template = None
    stops = []
    if "<|im_start|>" in chat and "<|im_end|>" in chat:
        template = ('{{ range .Messages }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n'
                    '{{ end }}<|im_start|>assistant\n')
        stops = ["<|im_end|>", "<|im_start|>"]
    elif "<|start_header_id|>" in chat and "<|eot_id|>" in chat:
        template = ('{{ range .Messages }}<|start_header_id|>{{ .Role }}<|end_header_id|>\n\n'
                    '{{ .Content }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>\n\n')
        stops = ["<|eot_id|>", "<|start_header_id|>"]
    eos = config.get("eos_token")
    if isinstance(eos, dict):
        eos = eos.get("content")
    if isinstance(eos, str) and eos and eos not in stops:
        stops.append(eos)
    lines = [f"FROM ./model-{quant}.gguf", "SYSTEM " + json.dumps(system, ensure_ascii=False)]
    if template:
        lines.append('TEMPLATE """' + template + '"""')
    lines.extend("PARAMETER stop " + json.dumps(stop, ensure_ascii=False) for stop in stops)
    return "\n".join(lines) + "\n", bool(template)


def zip_directory(source, destination):
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.name != ".complete":
                archive.write(path, path.relative_to(source))


def run_export(request, run):
    try:
        root = Path(run["dir"])
        if request.kind == "raw":
            source = Path(run["raw"])
        elif request.kind == "merged":
            source = merged_model(run)
        else:
            source_model = merged_model(run)
            converter = LLAMA / "convert_hf_to_gguf.py"
            quantizer = LLAMA / "build" / "bin" / "llama-quantize"
            if not converter.is_file() or (request.quant != "F16" and not quantizer.is_file()):
                raise FileNotFoundError("llama.cpp converter or llama-quantize is missing; rerun Cell 1.")
            source = root / ("ollama-" + uuid.uuid4().hex)
            source.mkdir()
            final = source / f"model-{request.quant}.gguf"
            intermediate = final if request.quant == "F16" else source / "intermediate-F16.gguf"
            if not Path(CONVERTER_PYTHON).is_file():
                raise FileNotFoundError("The llama.cpp converter environment is missing; rerun Cell 1.")
            run_command([CONVERTER_PYTHON, "-u", str(converter), str(source_model),
                         "--outfile", str(intermediate), "--outtype", "f16"])
            if request.quant != "F16":
                run_command([str(quantizer), str(intermediate), str(final), request.quant])
                intermediate.unlink()
            modelfile, known = make_modelfile(source_model, request.quant, request.system)
            (source / "Modelfile").write_text(modelfile, encoding="utf-8")
            notice = ("An automatically generated template is included; verify it against the original tokenizer.\n"
                      if known else
                      "No supported template was recognized. Ollama may use GGUF metadata; verify and add a model-specific TEMPLATE before chatting.\n")
            (source / "README.txt").write_text(
                "Extract all files together, then run:\nollama create my-custom-model -f Modelfile\n"
                "ollama run my-custom-model\n\n" + notice, encoding="utf-8")
            if not known:
                emit("WARNING: No known Ollama template recognized; review README.txt and the model's chat template.")
        archive = root / f"{request.kind}-{uuid.uuid4().hex}.zip"
        emit("Packaging ZIP (stored without compression)...")
        zip_directory(source, archive)
        with lock:
            state.update(phase="complete", message="Export ready", artifact=str(archive))
        emit(f"Export ready: {archive.name} ({archive.stat().st_size / 2**30:.2f} GiB)")
    except Exception as exc:
        fail(exc)


@app.post("/api/export")
def export(body: ExportRequest):
    with lock:
        run = state["run"]
        if not run:
            raise HTTPException(409, "Train and save a model before exporting.")
        begin("exporting")
        state["artifact"] = None
    threading.Thread(target=run_export, args=(body, run), daemon=True).start()
    return {"ok": True}


@app.get("/api/download")
def download():
    with lock:
        artifact = state["artifact"]
    if not artifact or not Path(artifact).is_file():
        raise HTTPException(404, "No export is ready.")
    return FileResponse(artifact, filename=Path(artifact).name, media_type="application/zip")


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Firefly Studio</title><script src="https://cdn.tailwindcss.com"></script>
<style>
body{background:#090c12;color:#e5e9f2;font-family:Inter,ui-sans-serif,system-ui,sans-serif}
.card{background:#111620;border:1px solid #252c3b;border-radius:14px;padding:22px}
input:not([type=checkbox]):not([type=file]),select,textarea{width:100%;background:#0b0f17;border:1px solid #30394a;border-radius:8px;color:#e8eefb;padding:9px 11px;outline:none}
input:focus,select:focus,textarea:focus{border-color:#8b7bff!important}
label{display:block;color:#a8b2c5;font-size:12px;font-weight:600;margin-bottom:6px}
.btn{border-radius:8px;padding:10px 15px;font-size:13px;font-weight:700;background:#252d3b;color:#e6eaff}
.btn:hover{filter:brightness(1.16)}.primary{background:#7969ed;color:white}.btn:disabled{opacity:.4;cursor:not-allowed}
.tag{border:1px solid #344057;border-radius:999px;padding:4px 10px;font-size:11px;color:#bdc9df}
.mono{font-family:ui-monospace,SFMono-Regular,monospace}
</style></head>
<body><header class="border-b border-slate-800 bg-[#0d111a]"><div class="max-w-[1500px] mx-auto px-6 py-5 flex flex-wrap items-center justify-between gap-3">
<div><div class="text-xl font-bold tracking-tight"><span class="text-violet-400">✦</span> FIREFLY <span class="text-slate-500 font-normal">/ fine-tuning studio</span></div><div class="text-xs text-slate-500 mt-1">Single-session model lab · Colab edition</div></div>
<div class="flex gap-2"><span id="hardware" class="tag">Detecting hardware</span><span id="phase" class="tag">CONNECTING</span></div></div></header>
<main class="max-w-[1500px] mx-auto px-6 py-7">
<div id="studio"><div id="warning" class="hidden mb-5 rounded-xl border border-amber-700/70 bg-amber-950/30 px-4 py-3 text-sm text-amber-200"></div>
<div class="grid grid-cols-1 xl:grid-cols-12 gap-5">
<section class="xl:col-span-4 space-y-5">
<div class="card"><div class="flex items-center justify-between mb-4"><h2 class="font-semibold">01 <span class="text-slate-400 ml-2">Model & access</span></h2><span id="hfUser" class="tag">Not connected</span></div>
<label>Hugging Face token <span class="font-normal text-slate-500">· optional for public models</span></label><div class="flex gap-2"><input id="hfToken" type="password" placeholder="hf_..." autocomplete="off"><button id="login" class="btn">Connect</button></div>
<div class="mt-4"><label>Model repository</label><input id="model" value="Qwen/Qwen2.5-0.5B-Instruct" placeholder="owner/model"></div>
<div class="flex items-center justify-between mt-3"><label class="!mb-0 flex gap-2 items-center"><input type="checkbox" id="trust"> Trust remote code</label><button id="inspect" class="text-xs text-violet-400 hover:text-violet-300">Inspect model ↗</button></div><p id="modelMeta" class="text-xs text-slate-500 mt-3">Inspect to estimate parameter count. Remote code can execute arbitrary code.</p></div>
<div class="card"><h2 class="font-semibold mb-4">02 <span class="text-slate-400 ml-2">Dataset</span></h2>
<label>Hugging Face dataset ID</label><input id="datasetId" placeholder="owner/dataset"><div class="mt-3"><label>Dataset split</label><input id="split" value="train"></div>
<div class="mt-4"><label>Or upload JSON / JSONL / Parquet</label><input id="file" type="file" accept=".json,.jsonl,.parquet" class="text-xs text-slate-400 w-full"></div><div id="uploadName" class="text-xs text-slate-500 mt-2">No file uploaded</div>
<button id="preview" class="btn mt-4 w-full">Preview formatted rows</button><pre id="previewRows" class="hidden mono whitespace-pre-wrap break-words max-h-60 overflow-auto rounded-lg bg-[#080b11] p-3 mt-3 text-xs text-slate-300"></pre>
<p class="text-xs text-slate-500 mt-3">Supports text, messages, prompt/completion, or instruction/output rows. Loss is computed over the entire formatted text.</p></div>
</section>
<section class="xl:col-span-4 space-y-5"><div class="card"><h2 class="font-semibold mb-4">03 <span class="text-slate-400 ml-2">Training configuration</span></h2>
<label>Fine-tuning method</label><select id="method"><option value="qlora">QLoRA · quantized base + adapters</option><option value="lora">LoRA · 16-bit base + adapters</option><option value="dora">DoRA · weight-decomposed adapters</option><option value="full">Full fine-tuning · all weights</option></select>
<div id="quantField" class="mt-4"><label>Base quantization</label><select id="bits"><option value="4">4-bit NF4 + double quantization</option><option value="8">8-bit</option></select></div>
<div id="adapterFields" class="grid grid-cols-3 gap-3 mt-4"><div><label>Rank</label><input id="rank" type="number" value="16" min="1"></div><div><label>Alpha</label><input id="alpha" type="number" value="32" min="1"></div><div><label>Dropout</label><input id="dropout" type="number" value="0.05" min="0" max="0.99" step="0.01"></div></div>
<div class="grid grid-cols-2 gap-3 mt-4"><div><label>Learning rate</label><input id="lr" type="number" value="0.0002" step="any"></div><div><label>Batch size</label><input id="batch" type="number" value="1" min="1"></div><div><label>Gradient accumulation</label><input id="accumulation" type="number" value="4" min="1"></div><div><label>Max sequence length</label><input id="length" type="number" value="1024" min="16"></div><div><label>Epochs</label><input id="epochs" type="number" value="1" min="0.01" step="0.1"></div><div><label>Max steps <span class="text-slate-600">(-1 = epochs)</span></label><input id="steps" type="number" value="-1" min="-1"></div></div>
<div class="mt-4"><label>Optimizer</label><select id="optimizer"><option value="adamw_torch">AdamW (PyTorch)</option><option value="paged_adamw_8bit">Paged AdamW 8-bit · CUDA only</option><option value="adafactor">Adafactor</option></select></div>
<label class="mt-4 flex gap-2 items-center"><input id="checkpointing" type="checkbox" checked> Gradient checkpointing</label><p id="estimate" class="text-xs text-slate-500 mt-3">Memory estimates are approximate and exclude activations and overhead.</p>
<div class="flex gap-2 mt-5"><button id="train" class="btn primary flex-1">▶ Start training</button><button id="stop" class="btn" disabled>■ Stop & save</button></div></div>
<div class="card"><h2 class="font-semibold mb-4">04 <span class="text-slate-400 ml-2">Export & quantize</span></h2>
<label>Output format</label><select id="exportKind"><option value="raw">Trained weights / adapters (.zip)</option><option value="merged">Merged full HF model (.zip)</option><option value="gguf">Ollama GGUF bundle (.zip)</option></select>
<div id="ggufFields" class="hidden"><div class="mt-4"><label>Quantization level</label><select id="quant"><option value="Q2_K">Q2_K · extreme compression</option><option value="Q3_K_M">Q3_K_M · small, lower quality</option><option value="Q4_K_M" selected>Q4_K_M · recommended balance</option><option value="Q5_K_M">Q5_K_M · high quality</option><option value="Q6_K">Q6_K · very high quality</option><option value="Q8_0">Q8_0 · high quality 8-bit</option><option value="F16">F16 · unquantized</option></select><p id="quantEstimate" class="text-xs text-slate-500 mt-2">Inspect model for a rough size estimate.</p></div>
<div class="mt-4"><label>Ollama system prompt</label><textarea id="system" rows="3">You are a helpful assistant.</textarea></div><p class="text-xs text-amber-300 mt-2">Only common ChatML and Llama 3 templates are generated. Verify template and stops for your model.</p></div>
<div class="flex gap-2 mt-4"><button id="export" class="btn flex-1">Build export</button><a id="download" class="btn hidden text-center" href="api/download">↓ Download</a></div><p class="text-xs text-slate-500 mt-3">Merging and ZIP creation require substantial CPU RAM and disk space.</p></div>
</section>
<section class="xl:col-span-4"><div class="card h-full"><div class="flex justify-between items-center"><h2 class="font-semibold">Live monitor</h2><span class="text-xs text-emerald-400">● STREAMING</span></div>
<div class="grid grid-cols-3 gap-3 mt-5"><div><div class="text-xs text-slate-500">STEP</div><div id="step" class="text-xl font-semibold mt-1">0 / —</div></div><div><div class="text-xs text-slate-500">LOSS</div><div id="loss" class="text-xl font-semibold mt-1">—</div></div><div><div class="text-xs text-slate-500">ETA</div><div id="eta" class="text-xl font-semibold mt-1">—</div></div></div>
<div class="h-1.5 bg-slate-800 rounded-full mt-5 overflow-hidden"><div id="progress" class="h-full bg-violet-400 transition-all" style="width:0%"></div></div><p id="message" class="text-xs text-slate-400 mt-2">Waiting for a job</p>
<div class="flex justify-between items-center mt-7 mb-2"><span class="text-xs uppercase tracking-wider text-slate-500">Terminal</span><span class="text-xs text-slate-600 mono">session.log</span></div><pre id="terminal" class="mono bg-[#080b11] rounded-lg border border-slate-800 p-4 text-xs text-slate-300 whitespace-pre-wrap break-words overflow-auto h-[450px] xl:h-[680px]"></pre></div></section>
</div><footer class="text-xs text-slate-600 mt-6">Artifacts and the Hugging Face login exist only in this runtime. Back up downloads before disconnecting. This studio has no built-in authentication; use a private, authenticated proxy and expose it only to trusted users.</footer></div></main>
<script>
const $=id=>document.getElementById(id);
const base=new URL('api/',location.href.endsWith('/')?location.href:location.href+'/');
let uploaded='', params=null, hw=null, phase='idle';
async function api(path,opts={}){
  const res=await fetch(new URL(path,base),opts);
  if(!res.ok){let msg;try{msg=(await res.json()).detail}catch{msg=await res.text()}throw Error(typeof msg==='string'?msg:JSON.stringify(msg))}
  return res.json();
}
const post=(path,data)=>api(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
function error(e){$('message').textContent=e.message; $('terminal').textContent+='\nERROR: '+e.message}
async function initialize(){
 try{const s=await api('status');hw=s.hardware;render(s);hardwareUI();inspectModel();stream()}
 catch(e){$('phase').textContent='OFFLINE';error(e)}
}
initialize();
function source(){return {model:$('model').value,trust_remote_code:$('trust').checked,dataset_id:$('datasetId').value.trim(),upload_id:uploaded,split:$('split').value}}
function hardwareUI(){
 $('hardware').textContent=hw.name+' · '+hw.precision.toUpperCase()+(hw.total_gib?' · '+hw.free_gib+'/'+hw.total_gib+' GiB free':'')+(hw.kind==='tpu'?' · '+hw.cores+' cores available':'');
 updateWarning();
}
function updateWarning(){
 let warnings=[];
 if(!hw)return;
 if(hw.kind==='cpu')warnings.push('CPU fallback: training may be extremely slow. QLoRA and paged AdamW require CUDA.');
 if(hw.kind==='tpu')warnings.push('TPU uses one XLA process/core with BF16; multi-core execution is not configured. QLoRA and paged AdamW are CUDA-only. Install a torch_xla build matching the Colab PyTorch runtime.');
 if(hw.warning)warnings.push(hw.warning);
 if(hw.kind==='cuda'&&hw.total_gib<=16&&$('method').value==='full'&&params>2e9)warnings.push('Full fine-tuning over 2B parameters on ≤16 GiB VRAM is likely to run out of memory. Use QLoRA or gradient checkpointing.');
 if(hw.precision==='fp16'&&$('method').value==='full')warnings.push('Full tuning uses FP32 master weights with FP16 mixed precision on this GPU; allow additional memory.');
 if(hw.kind!=='cuda'&&$('method').value==='qlora')warnings.push('Select LoRA, DoRA or full fine-tuning: QLoRA requires CUDA.');
 $('warning').textContent=warnings.join(' ');$('warning').classList.toggle('hidden',!warnings.length);
 if(params){let bits=$('method').value==='qlora'?Number($('bits').value):($('method').value==='full'&&hw.precision==='fp16'?32:(hw.precision==='fp32'?32:16));
 $('estimate').textContent='Approximate base weight memory: '+(params*bits/8/2**30).toFixed(2)+' GiB; excludes adapters, optimizer, activations and overhead.';
 const bpw={Q2_K:2.6,Q3_K_M:3.9,Q4_K_M:4.8,Q5_K_M:5.7,Q6_K:6.6,Q8_0:8.5,F16:16};
 $('quantEstimate').textContent='Rough tensor size: '+(params*bpw[$('quant').value]/8/2**30).toFixed(2)+' GiB; architecture and metadata alter actual size.'}
}
$('login').onclick=async()=>{try{let r=await post('login',{token:$('hfToken').value});$('hfToken').value='';$('hfUser').textContent=r.username}catch(e){error(e)}};
async function inspectModel(){let model=$('model').value.trim();if(!model)return;try{const r=await api('model?model='+encodeURIComponent(model));if(model!==$('model').value.trim())return;params=r.parameters||null;$('modelMeta').textContent=params?(params/1e9).toFixed(2)+'B parameters (Hub safetensors metadata)':'Parameter count unavailable in Hub safetensors metadata.';updateWarning()}catch(e){if(model!==$('model').value.trim())return;params=null;$('modelMeta').textContent=e.message;updateWarning()}}
$('inspect').onclick=inspectModel;
let inspectTimer;
$('model').oninput=()=>{params=null;$('modelMeta').textContent='Inspecting model metadata…';$('estimate').textContent='Memory estimates are approximate and exclude activations and overhead.';$('quantEstimate').textContent='Inspect model for a rough size estimate.';updateWarning();clearTimeout(inspectTimer);inspectTimer=setTimeout(inspectModel,1000)};
$('file').onchange=async()=>{if(!$('file').files.length)return;try{let fd=new FormData();fd.append('file',$('file').files[0]);let r=await api('upload',{method:'POST',body:fd});uploaded=r.upload_id;$('datasetId').value='';$('uploadName').textContent='Uploaded: '+r.filename}catch(e){uploaded='';error(e)}};
$('datasetId').oninput=()=>{if($('datasetId').value.trim()){uploaded='';$('file').value='';$('uploadName').textContent='Using Hub dataset'}};
$('preview').onclick=async()=>{try{$('previewRows').textContent='Loading…';$('previewRows').classList.remove('hidden');let r=await post('preview',source());$('previewRows').textContent=r.count+' rows\n\n'+r.rows.map((v,i)=>'['+(i+1)+'] '+v).join('\n\n')}catch(e){$('previewRows').textContent=e.message;error(e)}};
$('method').onchange=()=>{$('adapterFields').classList.toggle('hidden',$('method').value==='full');$('quantField').classList.toggle('hidden',$('method').value!=='qlora');$('lr').value=$('method').value==='full'?'0.00002':'0.0002';updateWarning()};
$('bits').onchange=updateWarning;$('quant').onchange=updateWarning;
$('exportKind').onchange=()=>$('ggufFields').classList.toggle('hidden',$('exportKind').value!=='gguf');
$('train').onclick=async()=>{try{await post('train',{...source(),method:$('method').value,bits:Number($('bits').value),rank:Number($('rank').value),alpha:Number($('alpha').value),dropout:Number($('dropout').value),learning_rate:Number($('lr').value),batch_size:Number($('batch').value),accumulation:Number($('accumulation').value),max_length:Number($('length').value),optimizer:$('optimizer').value,epochs:Number($('epochs').value),max_steps:Number($('steps').value),gradient_checkpointing:$('checkpointing').checked})}catch(e){error(e)}};
$('stop').onclick=async()=>{try{await post('stop',{})}catch(e){error(e)}};
$('export').onclick=async()=>{try{await post('export',{kind:$('exportKind').value,quant:$('quant').value,system:$('system').value})}catch(e){error(e)}};
function render(s){
 phase=s.phase;$('phase').textContent=phase.toUpperCase();$('message').textContent=s.message;$('step').textContent=s.step+' / '+(s.total||'—');
 $('loss').textContent=s.loss==null?'—':Number(s.loss).toFixed(4);$('eta').textContent=s.eta==null?'—':Math.ceil(s.eta/60)+'m';
 $('progress').style.width=(s.total?Math.min(100,100*s.step/s.total):0)+'%';
 const busy=phase==='training'||phase==='exporting';$('train').disabled=busy;$('export').disabled=busy;$('stop').disabled=phase!=='training';
 $('download').classList.toggle('hidden',!s.artifact||busy);$('download').href=new URL('download',base);
 let t=$('terminal'),follow=t.scrollTop+t.clientHeight>=t.scrollHeight-40;t.textContent=s.logs.join('\n');if(follow)t.scrollTop=t.scrollHeight;
}
async function stream(){
 while(true){try{let response=await fetch(new URL('events',base));if(!response.ok)throw Error('Stream connection failed');
 let reader=response.body.getReader(),decoder=new TextDecoder(),buffer='';
 while(true){let {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});let i;while((i=buffer.indexOf('\n\n'))>=0){let event=buffer.slice(0,i);buffer=buffer.slice(i+2);if(event.startsWith('data: '))render(JSON.parse(event.slice(6)))}}}
 catch(e){$('message').textContent='Connection interrupted; reconnecting…'}await new Promise(r=>setTimeout(r,2000))}
}
</script></body></html>'''
