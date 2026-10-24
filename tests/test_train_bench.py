"""CPU-safe tests for ``terser.hardware.train_bench`` (row 7.7)."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from terser.hardware import train_bench
from terser.hardware.train_bench import TrainResult, lora_target_modules, run_name
from terser.tracking.runs import RUN_NAME_RE


def test_run_names() -> None:
    names = [
        run_name("Qwen/Qwen2.5-0.5B-Instruct", "full", "fp32"),
        run_name("Qwen/Qwen2.5-1.5B-Instruct", "lora", "bf16"),
        run_name("Qwen/Qwen2.5-7B-Instruct", "qlora", "bf16"),
    ]
    assert names == [
        "7-trainbench-qwen2.50.5b-full_fp32-0",
        "7-trainbench-qwen2.51.5b-lora_r16-0",
        "7-trainbench-qwen2.57b-qlora_nf4-0",
    ]
    assert all(RUN_NAME_RE.match(n) for n in names)


def test_lora_targets_cover_attention_and_mlp() -> None:
    targets = lora_target_modules()
    assert {"q_proj", "v_proj", "down_proj"} <= set(targets)


def test_result_summary() -> None:
    result = TrainResult("m", "full", "fp32", 1024, 4, 10, 10, 1.0, 2.0, [0.5, 1.5])
    out = result.to_dict()
    assert out["mean_step_seconds"] == 1.0
    assert out["tokens_per_s"] == 4096.0
    assert TrainResult("m", "full", "fp32", 1024, 4, 1, 1, 0.0, 0.0).tokens_per_s == 0.0


def test_main_accepts_qlora(monkeypatch, capsys) -> None:
    result = TrainResult("m", "qlora", "bf16", 8, 1, 1, 2, 0.0, 0.0)
    run = Mock(return_value=result)
    monkeypatch.setattr(train_bench, "run", run)

    assert train_bench.main(["--model", "m", "--mode", "qlora", "--no-mlflow"]) == 0
    assert run.call_args.args[0].mode == "qlora"
    assert json.loads(capsys.readouterr().out)["mode"] == "qlora"


@pytest.fixture
def training_dependencies(monkeypatch):
    """Replace hardware and download boundaries; no tensor or device is created."""
    model = Mock()
    model.to.return_value = model
    model.config.vocab_size = 32
    model.parameters.return_value = [SimpleNamespace(requires_grad=True, numel=lambda: 4)]
    torch = SimpleNamespace(
        manual_seed=Mock(),
        device=Mock(return_value="cuda"),
        float32="float32",
        bfloat16="bfloat16",
        optim=SimpleNamespace(AdamW=Mock()),
        cuda=SimpleNamespace(
            reset_peak_memory_stats=Mock(),
            max_memory_allocated=Mock(return_value=0),
            max_memory_reserved=Mock(return_value=0),
        ),
    )
    transformers = SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
        BitsAndBytesConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
    )
    peft = SimpleNamespace(
        LoraConfig=Mock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs)),
        prepare_model_for_kbit_training=Mock(return_value=model),
        get_peft_model=Mock(return_value=model),
    )
    for name, module in (
        ("torch", torch),
        ("transformers", transformers),
        ("peft", peft),
        ("bitsandbytes", SimpleNamespace()),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return SimpleNamespace(model=model, torch=torch, transformers=transformers, peft=peft)


def bench_args(mode="qlora", weights_dtype="fp32"):
    return argparse.Namespace(
        model="local-test-model",
        mode=mode,
        weights_dtype=weights_dtype,
        seq_len=8,
        micro_batch=1,
        lora_r=8,
        warmup=0,
        steps=0,
    )


def test_qlora_quantization_adapters_and_logging(training_dependencies, monkeypatch) -> None:
    deps = training_dependencies
    result = train_bench.run(bench_args())

    deps.transformers.BitsAndBytesConfig.assert_called_once_with(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=deps.torch.bfloat16,
    )
    load_kwargs = deps.transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs
    assert load_kwargs["quantization_config"].bnb_4bit_quant_type == "nf4"
    assert load_kwargs["dtype"] == deps.torch.bfloat16
    assert load_kwargs["device_map"] == {"": "cuda"}
    deps.model.to.assert_not_called()
    deps.peft.prepare_model_for_kbit_training.assert_called_once_with(
        deps.model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    deps.peft.LoraConfig.assert_called_once_with(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    config = deps.peft.get_peft_model.call_args.args[1]
    deps.peft.get_peft_model.assert_called_once_with(deps.model, config)
    assert config.r == 8
    assert deps.model.config.use_cache is False
    quantization = {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
        "bnb_4bit_compute_dtype": "bf16",
    }
    assert json.loads(json.dumps(result.to_dict()))["quantization"] == quantization
    assert result.weights_dtype == "bf16"

    start = Mock(return_value=nullcontext())
    monkeypatch.setattr("terser.tracking.runs.start_run", start)
    monkeypatch.setattr("mlflow.log_metric", Mock())
    train_bench._log_mlflow(result)
    params = start.call_args.kwargs["params"]
    assert {key: params[key] for key in quantization} == quantization
    assert "qlora" in start.call_args.kwargs["run_name"]


def test_qlora_missing_bitsandbytes(training_dependencies, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    with pytest.raises(ImportError, match=r"QLoRA requires bitsandbytes.*terser\[gpu\]"):
        train_bench.run(bench_args())
    training_dependencies.transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()


@pytest.mark.parametrize("mode, dtype", [("full", "fp32"), ("full", "bf16"), ("lora", "bf16")])
def test_existing_modes_do_not_quantize(training_dependencies, monkeypatch, mode, dtype) -> None:
    monkeypatch.setitem(sys.modules, "bitsandbytes", None)
    deps = training_dependencies
    result = train_bench.run(bench_args(mode, dtype))
    deps.transformers.AutoModelForCausalLM.from_pretrained.assert_called_once_with(
        "local-test-model",
        dtype="float32" if dtype == "fp32" else "bfloat16",
    )
    deps.model.to.assert_called_once_with("cuda")
    deps.model.gradient_checkpointing_enable.assert_called_once_with(
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    deps.peft.prepare_model_for_kbit_training.assert_not_called()
    deps.transformers.BitsAndBytesConfig.assert_not_called()
    assert "quantization" not in result.to_dict()
