# Phase 2 smoke command ledger

Remote invocations used `ssh -F ~/.ssh/config <host> bash -s`; each linked `.sh` is the exact stdin payload, and each `.log` is actual combined output. Result files record exit status and wall time. Nested shell commands and serving flags are preserved in payloads and traced logs. Output below is trimmed to its final eight lines; full logs are linked. Failed attempts are retained and explained in the smoke report. All remote writes were confined to `~/terser` and `~/terser-env`.

## 01_install

[Exact command payload](../data/raw/p2_smoke_20261003/commands/01_install.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/01_install.log) · host `worker-1` · exit `130` · wall `1562.523` seconds

```text
 Downloaded nvidia-nccl-cu13
 Downloaded nvidia-cufft
 Downloaded nvidia-cudnn-cu13
 Downloaded nvidia-cusparselt-cu13
 Downloaded triton
 Downloaded nvidia-cusolver
 Downloaded vllm
Downloading torch (502.2MiB)
```

## 02_worker2_inventory

[Exact command payload](../data/raw/p2_smoke_20261003/commands/02_worker2_inventory.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/02_worker2_inventory.log) · host `worker-2` · exit `0` · wall `1.655` seconds

```text
|    0   N/A  N/A            1614      G   /usr/lib/xorg/Xorg                        9MiB |
|    0   N/A  N/A            1709      G   /usr/bin/gnome-shell                     10MiB |
+-----------------------------------------------------------------------------------------+
+ df -h ~
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  794G   97G  90% /
+ test -d ~/terser
+ true
```

## 03_install_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/03_install_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/03_install_progress.log) · host `worker-1` · exit `0` · wall `2.310` seconds

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  828G   63G  93% /
+ cd ~/terser
+ df -h .
+ du -sh .cache/uv ~/terser-env
2.7G	.cache/uv
92K	~/terser-env
```

## 04_disk_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/04_disk_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/04_disk_progress.log) · host `worker-1` · exit `0` · wall `1.682` seconds

```text
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  831G   60G  94% /
+ cd ~/terser
+ df -h .
+ du -sh .cache/uv
5.4G	.cache/uv
```

## 05_download

[Exact command payload](../data/raw/p2_smoke_20261003/commands/05_download.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/05_download.log) · host `worker-1` · exit `0` · wall `4064.142` seconds

```text
Fetching 11 files:  36%|███▋      | 4/11 [23:40<40:08, 344.12s/it]  
Fetching 11 files:  64%|██████▎   | 7/11 [23:40<09:14, 138.65s/it]
Fetching 11 files:  73%|███████▎  | 8/11 [23:40<05:24, 108.08s/it]
Fetching 11 files:  82%|████████▏ | 9/11 [23:41<02:44, 82.26s/it] 
Fetching 11 files:  91%|█████████ | 10/11 [23:43<01:01, 61.75s/it]
Fetching 11 files: 100%|██████████| 11/11 [29:11<00:00, 133.05s/it]
Fetching 11 files: 100%|██████████| 11/11 [29:11<00:00, 159.27s/it]
{"model": "Qwen/Qwen2.5-32B-Instruct-AWQ", "revision": "5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c", "download_bytes": 19338953966, "free_bytes_before": 36510236672, "path": "~/terser/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c", "wall_seconds": 1752.1623816380015, "free_bytes_after": 68953518080}
```

## 06_download_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/06_download_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/06_download_progress.log) · host `worker-1` · exit `0` · wall `2.451` seconds

```text
+ cd ~/terser
+ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  830G   61G  94% /
49M	.cache/huggingface
+ du -sh .cache/huggingface .cache/uv
4.6G	.cache/uv
```

## 07_network_probe

[Exact command payload](../data/raw/p2_smoke_20261003/commands/07_network_probe.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/07_network_probe.log) · host `worker-1` · exit `0` · wall `5.288` seconds

```text
access-control-expose-headers: *
content-length: 3945441440
vary: origin, access-control-request-method, access-control-request-headers
access-control-allow-origin: *
x-request-id: 01M407HK4G7EN7S1X4FQE67HNW
x-hf-cdn-pop: aws-ap-southeast-1
date: Sat, 03 Oct 2026 06:34:39 GMT

```

## 08_cache_status

[Exact command payload](../data/raw/p2_smoke_20261003/commands/08_cache_status.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/08_cache_status.log) · host `worker-1` · exit `0` · wall `2.319` seconds

```text
{"timestamp":"2026-10-03T06:34:52.739576Z","level":"INFO","fields":{"message":"Concurrency control for download: Decreased concurrency from 55 to 54; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)"},"filename":"/home/runner/work/xet-core/xet-core/xet_client/src/cas_client/adaptive_concurrency/controller.rs","line_number":586}
{"timestamp":"2026-10-03T06:34:53.242157Z","level":"INFO","fields":{"message":"Concurrency control for download: Decreased concurrency from 54 to 53; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)"},"filename":"/home/runner/work/xet-core/xet-core/xet_client/src/cas_client/adaptive_concurrency/controller.rs","line_number":586}
{"timestamp":"2026-10-03T06:34:53.746406Z","level":"INFO","fields":{"message":"Concurrency control for download: Decreased concurrency from 53 to 52; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)"},"filename":"/home/runner/work/xet-core/xet-core/xet_client/src/cas_client/adaptive_concurrency/controller.rs","line_number":586}
{"timestamp":"2026-10-03T06:34:54.260125Z","level":"INFO","fields":{"message":"Concurrency control for download: Decreased concurrency from 52 to 51; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)"},"filename":"/home/runner/work/xet-core/xet-core/xet_client/src/cas_client/adaptive_concurrency/controller.rs","line_number":586}
{"timestamp":"2026-10-03T06:34:54.762079Z","level":"INFO","fields":{"message":"Concurrency control for download: Decreased concurrency from 51 to 50; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)"},"filename":"/home/runner/work/xet-core/xet-core/xet_client/src/cas_client/adaptive_concurrency/controller.rs","line_number":586}
{"timestamp":"2026-10-03T06:34:55.284530Z","level":"INFO","fields":{"message":"Concurrency control for download: Decreased concurrency from 50 to 49; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)"},"filename":"/home/runner/work/xet-core/xet-core/xet_client/src/cas_client/adaptive_concurrency/controller.rs","line_number":586}
+ du -sh .cache/uv
4.6G	.cache/uv
```

## 09_sync_configs

[Full actual output](../data/raw/p2_smoke_20261003/commands/09_sync_configs.log)

```text
sending incremental file list

sent 628 bytes  received 15 bytes  428.67 bytes/sec
total size is 17,897  speedup is 27.83
```

## 09_sync_source

[Full actual output](../data/raw/p2_smoke_20261003/commands/09_sync_source.log)

```text
terser/tracking/__pycache__/runs.cpython-314.pyc
terser/utils/__pycache__/
terser/utils/__pycache__/__init__.cpython-314.pyc
terser/utils/__pycache__/dummy_experiment.cpython-314.pyc
terser/utils/__pycache__/seed.cpython-314.pyc

sent 1,908,117 bytes  received 3,758 bytes  44,985.29 bytes/sec
total size is 3,028,211  speedup is 1.58
```

## 10_transfer_probe

[Exact command payload](../data/raw/p2_smoke_20261003/commands/10_transfer_probe.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/10_transfer_probe.log) · host `worker-1` · exit `0` · wall `5.752` seconds

```text
+ cd ~/terser
+ curl -LsS --max-time 30 --range 0-1048575 -o .tmp/hf-probe.bin -w 'http=%{http_code} bytes=%{size_download} seconds=%{time_total} bytes_per_second=%{speed_download}\n' https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00001-of-00004.safetensors
http=206 bytes=1048576 seconds=3.564280 bytes_per_second=294190
```

## 11_transfer_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/11_transfer_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/11_transfer_progress.log) · host `worker-1` · exit `0` · wall `37.957` seconds

```text
67068111 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5.39ff5e83.incomplete
0 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185.e7fb60e0.incomplete
0 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd.a9aafe99.incomplete
+ du -sh .cache/huggingface .cache/uv
309M	.cache/huggingface
4.6G	.cache/uv
+ curl -LsS --max-time 45 --range 1048576-17825791 -o .tmp/hf-probe16.bin -w 'http=%{http_code} bytes=%{size_download} seconds=%{time_total} bytes_per_second=%{speed_download}\n' https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00001-of-00004.safetensors
http=206 bytes=16777216 seconds=35.165880 bytes_per_second=477087
```

## 12_worker2_transfer

[Exact command payload](../data/raw/p2_smoke_20261003/commands/12_worker2_transfer.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/12_worker2_transfer.log) · host `worker-2` · exit `0` · wall `8.361` seconds

```text
+ mkdir -p ~/terser/.tmp
+ cd ~/terser
+ curl -LsS --max-time 45 --range 1048576-17825791 -o .tmp/hf-probe16.bin -w 'http=%{http_code} bytes=%{size_download} seconds=%{time_total} bytes_per_second=%{speed_download}\n' https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/resolve/main/model-00001-of-00004.safetensors
http=206 bytes=16777216 seconds=6.980093 bytes_per_second=2403580
```

## 13_download_process

[Exact command payload](../data/raw/p2_smoke_20261003/commands/13_download_process.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/13_download_process.log) · host `worker-1` · exit `0` · wall `2.712` seconds

```text
1024000000 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185.e7fb60e0.incomplete
1024000000 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd.a9aafe99.incomplete
3.6G	.cache/huggingface
+ du -sh .cache/huggingface .cache/uv
5.0G	.cache/uv
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  834G   57G  94% /
+ df -h .
```

## 14_prepare

[Exact command payload](../data/raw/p2_smoke_20261003/commands/14_prepare.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/14_prepare.log) · host `worker-1` · exit `0` · wall `0.564` seconds

```text
+ cd ~/terser
+ PYTHONPATH=src
+ ./tools/download-env/bin/python tools/smoke_stage.py prepare
{"seed": 42, "ids": ["code-082", "code-015", "code-004", "code-036", "creative-032", "creative-029", "creative-018", "creative-014", "explain-070", "explain-012", "explain-076", "fact-055", "fact-005", "fact-004", "reason-012", "reason-028", "reason-030", "transform-065", "transform-078", "transform-004"], "categories": {"code": 4, "creative": 4, "explain": 3, "fact": 3, "reason": 3, "transform": 3}, "pii": {"clean": 20}}
+ sha256sum data/interim/p2_smoke_20261003/prompts.jsonl
620bae63a58ba7f0a5364316ef79c93ea8fac57e02c27be2d593d7de8571bac0  data/interim/p2_smoke_20261003/prompts.jsonl
```

## 15_versions_resolution

[Exact command payload](../data/raw/p2_smoke_20261003/commands/15_versions_resolution.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/15_versions_resolution.log) · host `worker-1` · exit `0` · wall `25.903` seconds

```text
1089958929 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd.a9aafe99.incomplete
4.0G	.cache/huggingface
+ find .cache/huggingface -name '*.incomplete' -printf '%s %p\n'
+ du -sh .cache/huggingface .cache/uv
4.2G	.cache/uv
+ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  833G   57G  94% /
```

## 16_transfer_diagnostics

[Exact command payload](../data/raw/p2_smoke_20261003/commands/16_transfer_diagnostics.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/16_transfer_diagnostics.log) · host `worker-1` · exit `0` · wall `1.801` seconds

```text
158M	.cache/uv/.tmpUQhnhW
13M	.cache/uv/.tmp22J6g2
44M	.cache/uv/.tmpmFpGeM
245M	.cache/uv/.tmpB3o5MC
14M	.cache/uv/.tmpPMPaCq
231M	.cache/uv/.tmpRhZ3UA
181M	.cache/uv/.tmpLzh9ZU
83M	.cache/uv/.tmp1XjcrK
```

## 17_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/17_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/17_progress.log) · host `worker-1` · exit `0` · wall `3.963` seconds

```text
991358092 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7.b6e2e275.incomplete
2153013816 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5.39ff5e83.incomplete
2128135882 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185.e7fb60e0.incomplete
2194675212 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd.a9aafe99.incomplete
+ ./tools/download-env/bin/python -
2026-10-03T06:45:17.446979Z Concurrency control for download: Decreased concurrency from 42 to 41; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)
2026-10-03T06:45:17.995063Z Concurrency control for download: Decreased concurrency from 41 to 40; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)
2026-10-03T06:45:18.500608Z Concurrency control for download: Decreased concurrency from 40 to 39; reason: success ratio below threshold (connection struggling) (success_ratio = 1.000, threshold = 0.500)
```

## 18_install_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/18_install_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/18_install_progress.log) · host `worker-1` · exit `0` · wall `2.525` seconds

```text
249M	.cache/uv/.tmpB3o5MC
76M	.cache/uv/.tmpPMPaCq
235M	.cache/uv/.tmpRhZ3UA
186M	.cache/uv/.tmpLzh9ZU
87M	.cache/uv/.tmp1XjcrK
    PID     ELAPSED %CPU %MEM COMMAND
1298393       16:56  3.4  0.0 ./tools/uv pip install --python ~/terser-env/bin/python vllm
+ ps -p 1298393 -o pid,etime,pcpu,pmem,args
```

## 19_bandwidth_isolation

[Exact command payload](../data/raw/p2_smoke_20261003/commands/19_bandwidth_isolation.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/19_bandwidth_isolation.log) · host `worker-1` · exit `0` · wall `46.804` seconds

```text
303700	.cache/uv/.tmpUQhnhW
41468	.cache/uv/.tmp22J6g2
112040	.cache/uv/.tmpmFpGeM
135312	.cache/uv/.tmpPMPaCq
471600	.cache/uv/.tmpRhZ3UA
129652	.cache/uv/.tmp1XjcrK
+ kill -CONT 1301708
+ trap - EXIT
```

## 20_install_transfer_window

[Exact command payload](../data/raw/p2_smoke_20261003/commands/20_install_transfer_window.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/20_install_transfer_window.log) · host `worker-1` · exit `0` · wall `121.774` seconds

```text
+ for i in $(seq 1 24)
+ kill -0 1298393
+ sleep 5
+ kill -CONT 1301708
+ trap - EXIT
+ du -sh .cache/uv ~/terser-env
7.3G	.cache/uv
92K	~/terser-env
```

## 21_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/21_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/21_progress.log) · host `worker-1` · exit `0` · wall `1.349` seconds

```text
7.3G	.cache/uv
92K	~/terser-env
490M	.cache/uv/.tmp3KBty5
+ find .cache/uv -maxdepth 2 -type d -name '.tmp*' -exec du -sh '{}' ';'
789M	.cache/uv/.tmpUQhnhW
+ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  840G   51G  95% /
```

## 22_smoke

[Exact command payload](../data/raw/p2_smoke_20261003/commands/22_smoke.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/22_smoke.log) · host `worker-1` · exit `1` · wall `2946.796` seconds

```text
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4312      G   /usr/lib/xorg/Xorg                        9MiB |
|    0   N/A  N/A            5114      G   /usr/bin/gnome-shell                     10MiB |
+-----------------------------------------------------------------------------------------+
+ exit 1
```

## 23_retry_install

[Exact command payload](../data/raw/p2_smoke_20261003/commands/23_retry_install.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/23_retry_install.log) · host `worker-1` · exit `0` · wall `385.546` seconds

```text
 + zipp==4.1.0
+ ~/terser-env/bin/python -c 'import sys,torch,vllm,transformers; print(sys.version); print("torch",torch.__version__,"cuda",torch.version.cuda,"vllm",vllm.__version__,"transformers",transformers.__version__); print(torch.cuda.get_device_name())'
3.11.17 (main, Oct  1 2026, 20:58:47) [Clang 22.1.3 ]
torch 2.13.0+cu130 cuda 13.0 vllm 0.30.0 transformers 5.18.0
NVIDIA GeForce RTX 4090
+ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  849G   42G  96% /
```

## 24_disk_after_runtime

[Exact command payload](../data/raw/p2_smoke_20261003/commands/24_disk_after_runtime.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/24_disk_after_runtime.log) · host `worker-1` · exit `0` · wall `1.578` seconds

```text
+ cd ~/terser
+ df -h .
+ du -sh .cache/huggingface .cache/uv ~/terser-env
8.2G	.cache/uv
32M	~/terser-env
+ find .cache/huggingface -name '*.incomplete' -printf '%s %p\n'
1024000000 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7.b6e2e275.incomplete
3603508960 .cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/blobs/f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185.e7fb60e0.incomplete
```

## 25_preload_baselines

[Exact command payload](../data/raw/p2_smoke_20261003/commands/25_preload_baselines.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/25_preload_baselines.log) · host `worker-1` · exit `0` · wall `321.084` seconds

```text
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
qwen2.5 6
llama-3.1 6

Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]
Loading weights: 100%|██████████| 199/199 [00:00<00:00, 14745.98it/s]
compressed_probe explain rain forms?
generator microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank?rate=0.5
```

## 26_dependency_check

[Exact command payload](../data/raw/p2_smoke_20261003/commands/26_dependency_check.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/26_dependency_check.log) · host `worker-1` · exit `0` · wall `10.870` seconds

```text
  VllmConfig              Dataclass which contains all vllm-related configuration. This
      simplifies passing around the distinct configurations in the codebase.
      

For full list:            vllm serve --help=all
For a section:            vllm serve --help=ModelConfig    (case-insensitive)
For a flag:               vllm serve --help=max-model-len  (_ or - accepted)
Documentation:            https://docs.vllm.ai
```

## 27_download_checkpoint

[Exact command payload](../data/raw/p2_smoke_20261003/commands/27_download_checkpoint.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/27_download_checkpoint.log) · host `worker-1` · exit `0` · wall `2.316` seconds

```text
/dev/nvme0n1p2  938G  850G   41G  96% /
16G	.cache/huggingface
+ df -h .
+ du -sh .cache/huggingface
+ find .cache/huggingface -name '*.incomplete' -printf '%s %p\n'
0 .cache/huggingface/hub/models--microsoft--llmlingua-2-bert-base-multilingual-cased-meetingbank/blobs/22b9ecde52fec5c97e8c54a293be768727df95a81c6c8dccb03f262a50c58324.c3079d7b.incomplete
134152910 .cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/blobs/41246eed2b75b66526339c5d32d6f7acdefe0bd24180f97c74303f4656877344.7b2e2df6.incomplete
1046305666 .cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/blobs/c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1e6c1c6a95497d7a8837.a5197d92.incomplete
```

## 28_copy_versions

[Full actual output](../data/raw/p2_smoke_20261003/commands/28_copy_versions.log)

```text
receiving incremental file list
versions.json

sent 43 bytes  received 458 bytes  143.14 bytes/sec
total size is 346  speedup is 0.69
```

## 29_download_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/29_download_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/29_download_progress.log) · host `worker-1` · exit `0` · wall `1.567` seconds

```text
Saturday 03 October 2026 07:11:31 AM UTC
+ cd ~/terser
+ date -u
+ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  856G   35G  97% /
+ find .cache/huggingface -name '*.incomplete' -printf '%s %p\n'
4746138976 .cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/blobs/c5214cdb995ed3dd716add8d9efbfe016b76bb2f1c4c1e6c1c6a95497d7a8837.a5197d92.incomplete
```

## 30_code_snapshot

[Exact command payload](../data/raw/p2_smoke_20261003/commands/30_code_snapshot.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/30_code_snapshot.log) · host `worker-1` · exit `0` · wall `2.853` seconds

```text
+ tar --exclude=__pycache__ '--exclude=*.egg-info' -czf data/raw/p2_smoke_20261003/code_snapshot.tar.gz src configs pyproject.toml dvc.yaml params.yaml
+ sha256sum data/raw/p2_smoke_20261003/code_snapshot.tar.gz
9fbe0e454cf6394a5933fda0215c6b09c9596e6ba500545e384de6c1f11e1476  data/raw/p2_smoke_20261003/code_snapshot.tar.gz
+ sha256sum src/terser/data/pii.py src/terser/data/candidates.py src/terser/phase2/respond.py src/terser/phase2/label_stage.py
7cc07684b2a27c598d88675cb82cd60622389c97680cf105e86ea56891e10ab9  src/terser/data/pii.py
07e13c24b2c667b7cd87ee559926db761d28e97e987a4e4b7c17866606a0619f  src/terser/data/candidates.py
f8a0aeab5239cfb0a8cd0f67d932f2a08d20f5af79d291b49822dbd5ff6c0d29  src/terser/phase2/respond.py
78768136f7b96efb8571db9749ae9d4ee7fa735c5a18140f97d890b8cc5de06c  src/terser/phase2/label_stage.py
```

## 31_judge_download_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/31_judge_download_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/31_judge_download_progress.log) · host `worker-1` · exit `0` · wall `1.810` seconds

```text
+ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  860G   30G  97% /
+ find .cache/huggingface -path '*32B*' -name '*.incomplete' -printf '%s %p\n'
1072743290 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/ece82d2cd5ecc9691572aae55cf2d66fa52edd2c199d5a6dccb8182145bae59c.9fe3174d.incomplete
1074855145 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/548f5c7078e297088c74bec4443bfe9e3b4183ee7457f328107c37a5eb861ea1.58cf94ae.incomplete
1207115231 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/f158db037c405677ea6ca21a5cb67a800ddc256217c722c0dee0eb31f3f75fb8.4ab8bb1f.incomplete
1073031561 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/735d1b5aa8ef01420f2079b355d84cfa7a4b37571d44715276b3b903c06d65d8.eedaa802.incomplete
```

## 32_judge_download_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/32_judge_download_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/32_judge_download_progress.log) · host `worker-1` · exit `0` · wall `1.987` seconds

```text
+ df -h .
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  865G   26G  98% /
2212862275 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/ece82d2cd5ecc9691572aae55cf2d66fa52edd2c199d5a6dccb8182145bae59c.9fe3174d.incomplete
2147716236 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/548f5c7078e297088c74bec4443bfe9e3b4183ee7457f328107c37a5eb861ea1.58cf94ae.incomplete
2280360931 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/f158db037c405677ea6ca21a5cb67a800ddc256217c722c0dee0eb31f3f75fb8.4ab8bb1f.incomplete
2280280777 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/735d1b5aa8ef01420f2079b355d84cfa7a4b37571d44715276b3b903c06d65d8.eedaa802.incomplete
+ find .cache/huggingface -path '*32B*' -name '*.incomplete' -printf '%s %p\n'
```

## 33_judge_download_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/33_judge_download_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/33_judge_download_progress.log) · host `worker-1` · exit `0` · wall `1.377` seconds

```text
Saturday 03 October 2026 07:40:07 AM UTC
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p2  938G  826G   65G  93% /
+ cd ~/terser
+ date -u
+ df -h .
+ find .cache/huggingface -path '*32B*' -name '*.incomplete' -printf '%s %p\n'
2930717549 .cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/blobs/1af3bf12ce80cf2f85bacb57a7d2fd584712a945f481cf394f0c7b17983269e5.e66da9a3.incomplete
```

## 34_startup_failure

[Exact command payload](../data/raw/p2_smoke_20261003/commands/34_startup_failure.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/34_startup_failure.log) · host `worker-1` · exit `0` · wall `0.796` seconds

```text
(APIServer pid=1355336)     with launch_core_engines(
(APIServer pid=1355336)   File "~/terser/.python/cpython-3.11.17-linux-x86_64-gnu/lib/python3.11/contextlib.py", line 144, in __exit__
(APIServer pid=1355336)     next(self.gen)
(APIServer pid=1355336)   File "~/terser-env/lib/python3.11/site-packages/vllm/v1/engine/utils.py", line 1240, in launch_core_engines
(APIServer pid=1355336)     wait_for_engine_startup(
(APIServer pid=1355336)   File "~/terser-env/lib/python3.11/site-packages/vllm/v1/engine/utils.py", line 1320, in wait_for_engine_startup
(APIServer pid=1355336)     raise RuntimeError(
(APIServer pid=1355336) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

## 35_kv_block_requirement

[Exact command payload](../data/raw/p2_smoke_20261003/commands/35_kv_block_requirement.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/35_kv_block_requirement.log) · host `worker-1` · exit `0` · wall `0.979` seconds

```text
900:             f"To serve at least one request with the model's max seq len "
901:             f"({max_model_len}), ({format_gib(needed_memory)} GiB KV "
902:             f"cache is needed, which is larger than the available KV cache "
903:             f"memory ({format_gib(available_memory)} GiB). {estimated_msg}"
904:             f"Try increasing `gpu_memory_utilization` (which also controls "
905:             f"CPU memory on the CPU backend) or decreasing `max_model_len` "
kv_bytes_per_token 262144 extra_16_token_block_bytes 4194304 revised_kv_cap_bytes 1077936128
+ cp gpu_queue/p2-candidates-qwen32b.yaml data/raw/p2_smoke_20261003/queued_config.yaml
```

## 36_smoke_retry

[Exact command payload](../data/raw/p2_smoke_20261003/commands/36_smoke_retry.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/36_smoke_retry.log) · host `worker-1` · exit `1` · wall `32.433` seconds

```text
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4312      G   /usr/lib/xorg/Xorg                        9MiB |
|    0   N/A  N/A            5114      G   /usr/bin/gnome-shell                     10MiB |
+-----------------------------------------------------------------------------------------+
+ exit 1
```

## 37_fixed_startup_failure

[Exact command payload](../data/raw/p2_smoke_20261003/commands/37_fixed_startup_failure.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/37_fixed_startup_failure.log) · host `worker-1` · exit `0` · wall `1.290` seconds

```text
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366]  /usr/bin/nvcc --generate-dependencies-with-compile -MF ~/terser/.cache/flashinfer/.cache/flashinfer/0.6.18.post1/89/cached_ops/sampling/csrc_renorm.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/cub -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/libcudacxx/include -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/thrust -isystem ~/terser/.python/cpython-3.11.17-linux-x86_64-gnu/include/python3.11 -isystem ~/terser-env/lib/python3.11/site-packages/tvm_ffi/include -isystem ~/terser-env/lib/python3.11/site-packages/tvm_ffi/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/csrc -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/cutlass/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/cutlass/tools/util/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all --compress-mode=size -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c ~/terser-env/lib/python3.11/site-packages/flashinfer/data/csrc/renorm.cu -o ~/terser/.cache/flashinfer/.cache/flashinfer/0.6.18.post1/89/cached_ops/sampling/csrc_renorm.cuda.o 
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366] nvcc fatal   : Unknown option '--compress-mode=size'
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366] [3/4]  /usr/bin/nvcc --generate-dependencies-with-compile -MF ~/terser/.cache/flashinfer/.cache/flashinfer/0.6.18.post1/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/cub -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/libcudacxx/include -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/thrust -isystem ~/terser/.python/cpython-3.11.17-linux-x86_64-gnu/include/python3.11 -isystem ~/terser-env/lib/python3.11/site-packages/tvm_ffi/include -isystem ~/terser-env/lib/python3.11/site-packages/tvm_ffi/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/csrc -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/cutlass/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/cutlass/tools/util/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all --compress-mode=size -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c ~/terser-env/lib/python3.11/site-packages/flashinfer/data/csrc/flashinfer_sampling_binding.cu -o ~/terser/.cache/flashinfer/.cache/flashinfer/0.6.18.post1/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o 
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366] FAILED: [code=1] ~/terser/.cache/flashinfer/.cache/flashinfer/0.6.18.post1/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o 
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366]  /usr/bin/nvcc --generate-dependencies-with-compile -MF ~/terser/.cache/flashinfer/.cache/flashinfer/0.6.18.post1/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o.d -DPy_LIMITED_API=0x03090000 -D_GLIBCXX_USE_CXX11_ABI=1 -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/cub -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/libcudacxx/include -I~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/thrust -isystem ~/terser/.python/cpython-3.11.17-linux-x86_64-gnu/include/python3.11 -isystem ~/terser-env/lib/python3.11/site-packages/tvm_ffi/include -isystem ~/terser-env/lib/python3.11/site-packages/tvm_ffi/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/csrc -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/cutlass/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/cutlass/tools/util/include -isystem ~/terser-env/lib/python3.11/site-packages/flashinfer/data/spdlog/include --compiler-options=-fPIC --expt-relaxed-constexpr -gencode=arch=compute_89,code=sm_89 -DFLASHINFER_ENABLE_FP8_E8M0 -DFLASHINFER_ENABLE_FP4_E2M1 -std=c++17 --threads=1 -use_fast_math -Xfatbin=-compress-all --compress-mode=size -DFLASHINFER_ENABLE_F16 -DFLASHINFER_ENABLE_BF16 -DFLASHINFER_ENABLE_FP8_E4M3 -DFLASHINFER_ENABLE_FP8_E5M2 -DNDEBUG -O3 -c ~/terser-env/lib/python3.11/site-packages/flashinfer/data/csrc/flashinfer_sampling_binding.cu -o ~/terser/.cache/flashinfer/.cache/flashinfer/0.6.18.post1/89/cached_ops/sampling/csrc_flashinfer_sampling_binding.cuda.o 
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366] nvcc fatal   : Unknown option '--compress-mode=size'
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366] ninja: build stopped: subcommand failed.
(EngineCore pid=1358421) ERROR 10-03 13:13:48 [core.py:1366] 
```

## 38_nvcc_resolution

[Exact command payload](../data/raw/p2_smoke_20261003/commands/38_nvcc_resolution.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/38_nvcc_resolution.log) · host `worker-1` · exit `0` · wall `1.092` seconds

```text
Copyright (c) 2005-2021 NVIDIA Corporation
Built on Thu_Nov_18_09:45:30_PST_2021
Cuda compilation tools, release 11.5, V11.5.119
Build cuda_11.5.r11.5/compiler.30672275_0
+ ~/terser-env/bin/python -
nvidia-cuda-nvcc ~/terser-env/lib/python3.11/site-packages/nvidia/cu13/bin/nvcc
nvidia-cuda-runtime ~/terser-env/lib/python3.11/site-packages/nvidia/cu13/include/cuda_runtime.h
nvidia-cuda-runtime ~/terser-env/lib/python3.11/site-packages/nvidia/cu13/lib/libcudart.so.13
```

## 39_smoke_cuda

[Exact command payload](../data/raw/p2_smoke_20261003/commands/39_smoke_cuda.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/39_smoke_cuda.log) · host `worker-1` · exit `1` · wall `52.271` seconds

```text
++ cat logs/candidates32cuda_server.pid
+ pid=1360761
+ kill -0 1360761
++ cat logs/candidates32cuda_sampler.pid
+ kill 1360759
+ date +%s.%N
+ nvidia-smi
+ exit 1
```

## 40_cuda_startup_failure

[Exact command payload](../data/raw/p2_smoke_20261003/commands/40_cuda_startup_failure.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/40_cuda_startup_failure.log) · host `worker-1` · exit `0` · wall `1.387` seconds

```text
(EngineCore pid=1361047)    41 | #      error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
(EngineCore pid=1361047)       |        ^~~~~
(EngineCore pid=1361047)                  from ~/terser-env/lib/python3.11/site-packages/flashinfer/data/csrc/renorm.cu:16:
(EngineCore pid=1361047) ~/terser-env/lib/python3.11/site-packages/flashinfer/data/cccl/libcudacxx/include/cuda/std/__cccl/cuda_toolkit.h:41:8: error: #error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
(EngineCore pid=1361047)    41 | #      error "CUDA compiler and CUDA toolkit headers are incompatible, please check your include paths"
(EngineCore pid=1361047)       |        ^~~~~
(APIServer pid=1360761)     raise RuntimeError(
(APIServer pid=1360761) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
```

## 41_cuda_package_versions

[Exact command payload](../data/raw/p2_smoke_20261003/commands/41_cuda_package_versions.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/41_cuda_package_versions.log) · host `worker-1` · exit `0` · wall `1.990` seconds

```text
+ cd ~/terser
+ ~/terser-env/bin/python -
nvidia-cuda-nvcc ['13.0.48', '13.0.88']
nvidia-cuda-crt ['13.0.48', '13.0.88']
nvidia-nvvm ['13.0.48', '13.0.88']
```

## 42_align_cuda_compiler

[Exact command payload](../data/raw/p2_smoke_20261003/commands/42_align_cuda_compiler.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/42_align_cuda_compiler.log) · host `worker-1` · exit `0` · wall `10.090` seconds

```text
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Aug_20_01:58:59_PM_PDT_2025
Cuda compilation tools, release 13.0, V13.0.88
Build cuda_13.0.r13.0/compiler.36424714_0
+ ./tools/uv pip check --python ~/terser-env/bin/python
Using Python 3.11.17 environment at: ~/terser-env
Checked 328 packages in 2ms
All installed packages are compatible
```

## 43_record_startup_failures

[Exact command payload](../data/raw/p2_smoke_20261003/commands/43_record_startup_failures.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/43_record_startup_failures.log) · host `worker-1` · exit `1` · wall `5.512` seconds

```text
+ ~/terser-env/bin/python tools/record_startup_failures.py
2026/10/03 13:20:59 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/10/03 13:20:59 INFO mlflow.store.db.utils: Updating database tables
Traceback (most recent call last):
  File "~/terser/tools/record_startup_failures.py", line 12, in <module>
    with (ROOT/'logs'/f'{log_name}_baseline.csv').open() as f: baseline=int(next(csv.DictReader(f))['memory.used [MiB]'].split()[0])
                                                                            ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^
KeyError: 'memory.used [MiB]'
```

## 44_smoke_aligned

[Exact command payload](../data/raw/p2_smoke_20261003/commands/44_smoke_aligned.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/44_smoke_aligned.log) · host `worker-1` · exit `1` · wall `320.673` seconds

```text
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4312      G   /usr/lib/xorg/Xorg                        9MiB |
|    0   N/A  N/A            5114      G   /usr/bin/gnome-shell                     10MiB |
+-----------------------------------------------------------------------------------------+
+ exit 1
```

## 45_sampling_header

[Exact command payload](../data/raw/p2_smoke_20261003/commands/45_sampling_header.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/45_sampling_header.log) · host `worker-1` · exit `0` · wall `0.662` seconds

```text
timestamp, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
2026/10/03 13:10:50.137, 43 MiB, 24564 MiB, 0 %
timestamp, memory.used [MiB], memory.total [MiB], utilization.gpu [%]
2026/10/03 13:10:50.154, 43 MiB, 24564 MiB, 0 %
2026/10/03 13:10:50.355, 43 MiB, 24564 MiB, 0 %
+ cd ~/terser
+ head -3 logs/candidates32_baseline.csv
+ head -3 logs/candidates32_vram.csv
```

## 46_record_startup_failures

[Exact command payload](../data/raw/p2_smoke_20261003/commands/46_record_startup_failures.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/46_record_startup_failures.log) · host `worker-1` · exit `0` · wall `3.021` seconds

```text
      "peak_net_vram_gib": 22.388671875,
      "generated_tokens": 0,
      "server_reserved_gpu_hours": 0.013910103572739495,
      "sample_count": 251,
      "sample_interval_ms": 200
    }
  }
]
```

## 47_judge_resume

[Exact command payload](../data/raw/p2_smoke_20261003/commands/47_judge_resume.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/47_judge_resume.log) · host `worker-1` · exit `0` · wall `297.908` seconds

```text
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            4312      G   /usr/lib/xorg/Xorg                        9MiB |
|    0   N/A  N/A            5114      G   /usr/bin/gnome-shell                     10MiB |
+-----------------------------------------------------------------------------------------+
+ trap - EXIT
SMOKE COMPLETE; no full-scale run started
+ printf 'SMOKE COMPLETE; no full-scale run started\n'
```

## 48_judge_progress

[Exact command payload](../data/raw/p2_smoke_20261003/commands/48_judge_progress.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/48_judge_progress.log) · host `worker-1` · exit `0` · wall `0.917` seconds

```text
(APIServer pid=1374125) INFO:     Shutting down
(APIServer pid=1374125) INFO:     Waiting for application shutdown.
(APIServer pid=1374125) INFO:     Application shutdown complete.
~/terser/.python/cpython-3.11.17-linux-x86_64-gnu/lib/python3.11/multiprocessing/resource_tracker.py:254: UserWarning: resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  warnings.warn('resource_tracker: There appear to be %d '
+ ~/terser-env/bin/python -
cached_response_files 868
judge_logprob_responses 494
```

## 49_verify

[Exact command payload](../data/raw/p2_smoke_20261003/commands/49_verify.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/49_verify.log) · host `worker-1` · exit `1` · wall `1.585` seconds

```text
+ HF_HOME=~/terser/.cache/huggingface
+ TMPDIR=~/terser/.tmp
+ ~/terser-env/bin/python tools/verify_and_summarize.py
Traceback (most recent call last):
  File "~/terser/tools/verify_and_summarize.py", line 39, in <module>
    assert actual==rec['download_bytes']
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
```

## 50_snapshot_sizes

[Exact command payload](../data/raw/p2_smoke_20261003/commands/50_snapshot_sizes.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/50_snapshot_sizes.log) · host `worker-1` · exit `0` · wall `1.090` seconds

```text
actual_snapshot_bytes 15242788168
[('model.safetensors.index.json', 27752), ('model-00004-of-00004.safetensors', 3556377672), ('model-00002-of-00004.safetensors', 3864726352), ('tokenizer_config.json', 7305), ('tokenizer.json', 7031645), ('generation_config.json', 243), ('model-00003-of-00004.safetensors', 3864726424), ('vocab.json', 2776833), ('config.json', 663), ('merges.txt', 1671839), ('model-00001-of-00004.safetensors', 3945441440)]
{'model': 'microsoft/Phi-3.5-mini-instruct', 'revision': '2fe192450127e6a83f7441aef6e3ca586c338b77', 'download_bytes': 7644642041, 'free_bytes_before': 45397053440, 'path': '~/terser/.cache/huggingface/hub/models--microsoft--Phi-3.5-mini-instruct/snapshots/2fe192450127e6a83f7441aef6e3ca586c338b77', 'wall_seconds': 664.1917838969966, 'free_bytes_after': 36510236672}
actual_snapshot_bytes 7644642041
[('model-00001-of-00002.safetensors', 4972489328), ('model-00002-of-00002.safetensors', 2669692552), ('configuration_phi3.py', 11153), ('model.safetensors.index.json', 16331), ('tokenizer.model', 499723), ('special_tokens_map.json', 665), ('tokenizer_config.json', 3984), ('modeling_phi3.py', 73814), ('tokenizer.json', 1844408), ('generation_config.json', 195), ('added_tokens.json', 306), ('config.json', 3451), ('sample_finetune.py', 6131)]
{'model': 'Qwen/Qwen2.5-32B-Instruct-AWQ', 'revision': '5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c', 'download_bytes': 19338953966, 'free_bytes_before': 36510236672, 'path': '~/terser/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct-AWQ/snapshots/5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c', 'wall_seconds': 1752.1623816380015, 'free_bytes_after': 68953518080}
actual_snapshot_bytes 19340625805
[('model-00004-of-00005.safetensors', 3980134240), ('model-00005-of-00005.safetensors', 3477738728), ('model.safetensors.index.json', 143195), ('model-00003-of-00005.safetensors', 3947411392), ('tokenizer_config.json', 7305), ('tokenizer.json', 7031645), ('generation_config.json', 243), ('vocab.json', 2776833), ('config.json', 841), ('model-00001-of-00005.safetensors', 3943575336), ('merges.txt', 1671839), ('model-00002-of-00005.safetensors', 3980134208)]
```

## 51_verify

[Exact command payload](../data/raw/p2_smoke_20261003/commands/51_verify.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/51_verify.log) · host `worker-1` · exit `0` · wall `2.481` seconds

```text
VERIFIED: prompt/candidate/response/count/label coverage, real calls, creative seeds, tracking tags and sampled VRAM
name, memory.used [MiB]
NVIDIA GeForce RTX 4090, 43 MiB
+ nvidia-smi --query-gpu=name,memory.used --format=csv
pid, process_name, used_gpu_memory [MiB]
+ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
+ curl -fsS http://127.0.0.1:8012/v1/models
curl: (7) Failed to connect to 127.0.0.1 port 8012 after 0 ms: Connection refused
```

## 52_final_inventory

[Exact command payload](../data/raw/p2_smoke_20261003/commands/52_final_inventory.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/52_final_inventory.log) · host `worker-1` · exit `1` · wall `0.821` seconds

```text
+ cd ~/terser
+ export HF_HOME=~/terser/.cache/huggingface TMPDIR=~/terser/.tmp
+ HF_HOME=~/terser/.cache/huggingface
+ TMPDIR=~/terser/.tmp
+ ~/terser-env/bin/python -m pip freeze
~/terser-env/bin/python: No module named pip
```

## 53_final_inventory

[Exact command payload](../data/raw/p2_smoke_20261003/commands/53_final_inventory.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/53_final_inventory.log) · host `worker-1` · exit `0` · wall `2.380` seconds

```text
Filesystem         1B-blocks         Used   Available Use% Mounted on
/dev/nvme0n1p2 1006684778496 887051890688 68420648960  93% /
name, driver_version, memory.total [MiB], memory.used [MiB]
NVIDIA GeForce RTX 4090, 595.84, 24564 MiB, 43 MiB
+ df -B1 ~/terser
+ nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used --format=csv
+ nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
pid, process_name, used_gpu_memory [MiB]
```

## 54_copy_final

[Full actual output](../data/raw/p2_smoke_20261003/commands/54_copy_final.log)

```text
receiving incremental file list
./
mlflow.db
mlflow_all_runs.json
pip_freeze.txt

sent 6,278 bytes  received 126,846 bytes  53,249.60 bytes/sec
total size is 35,627,471  speedup is 267.63
```

## 54_copy_interim

[Full actual output](../data/raw/p2_smoke_20261003/commands/54_copy_interim.log)

```text
labels.jsonl
prompts.jsonl
responses_phi35.jsonl
responses_qwen7b.jsonl
token_counts.parquet

sent 141 bytes  received 1,130,841 bytes  753,988.00 bytes/sec
total size is 1,130,068  speedup is 1.00
```

## 54_copy_raw

[Full actual output](../data/raw/p2_smoke_20261003/commands/54_copy_raw.log)

```text
response_cache/ff/
response_cache/ff/ff38cf6b02eff126b136342ba17809db02c25965d2adb6ca9eb123c4354033c4.json
response_cache/ff/ffa98f51085ff1b4c151332e561935e27c7cd5958e93acf4754e63ccbd50bce0.json
response_cache/ff/ffbb4d333de253f11de360a43c73aa1407deb75c238a0fb8e61b5dcb17c8a300.json
response_cache/ff/ffd6216ab6d89f7c84c7233c78f5630abd3561f085d0e142e06b7e200fa53f64.json

sent 19,292 bytes  received 35,732,840 bytes  7,944,918.22 bytes/sec
total size is 35,605,359  speedup is 1.00
```

## 54_copy_server_logs

[Full actual output](../data/raw/p2_smoke_20261003/commands/54_copy_server_logs.log)

```text
qwen7_server.log
qwen7_server.pid
qwen7_start_seconds.txt
qwen7_stop_seconds.txt
qwen7_vram.csv

sent 1,554 bytes  received 550,085 bytes  220,655.60 bytes/sec
total size is 543,855  speedup is 0.99
```

## 55_tracking_failed_wrapper

[Exact command payload](../data/raw/p2_smoke_20261003/commands/55_tracking_failed_wrapper.sh) · [Full actual output](../data/raw/p2_smoke_20261003/commands/55_tracking_failed_wrapper.log) · host `worker-1` · exit `0` · wall `2.321` seconds

```text
229be517a78a4483bfd087e78b4a3b51 p2-judge-qwen32b-providererror-s42 FAILED
57aa35c9fcc542f89fb20839e032124b p2-respond-phi3.5mini-smoke-s42 FINISHED
6f6aa31b52dd4ecc8032ca041d6ebf79 p2-respond-qwen7b-smoke-s42 FINISHED
be1268cb75ee4fc0895e968be2a2c729 p2-counts-qwen7b-smoke-s42 FINISHED
9956ec544f2d41c29f8ffa3cbefb36a0 p2-candidates-qwen32b-smoke-s42 FINISHED
39818b3379bc41d8b74a545f4b1e1915 p7-serving-qwen32b-toolkit-s42 FAILED
25610f89871b47d491f921e39c26cc8f p7-serving-qwen32b-kvblock-s42 FAILED
fae1068486234367b9ac69216905d711 p7-serving-qwen32b-queue-s42 FAILED
```

## 56_copy_tracking

[Full actual output](../data/raw/p2_smoke_20261003/commands/56_copy_tracking.log)

```text
receiving incremental file list
mlflow.db
mlflow_all_runs.json

sent 6,122 bytes  received 40,242 bytes  30,909.33 bytes/sec
total size is 978,430  speedup is 21.10
```

## Local copy and tracking portability

The following copy commands completed; their transfer output is retained as `54_copy_*.log` above. Source/config and wrapper uploads used the same SSH configuration and rsync transport; individual upload output is also present in the interactive task transcript.

```sh
rsync -av -e "ssh -F $HOME/.ssh/config" worker-1:terser/data/raw/p2_smoke_20261003/ data/raw/p2_smoke_20261003/
rsync -av -e "ssh -F $HOME/.ssh/config" worker-1:terser/data/interim/p2_smoke_20261003/ data/interim/p2_smoke_20261003/
rsync -av -e "ssh -F $HOME/.ssh/config" --include='candidates32*' --include='qwen7*' --include='phi35*' --include='judge32*' --exclude='*' worker-1:terser/logs/ data/raw/p2_smoke_20261003/server_logs/
```

SQLite was copied before rebasing `experiments.artifact_location` and `runs.artifact_uri` from the worker prefix to the local raw-data directory. `PRAGMA integrity_check` returned `ok`. The worker database backup remains unchanged. No model weights were copied locally. Local Git staging failed with a read-only filesystem error; no commit or push was performed. DVC and local verification output follow in the linked handoff.

Local DVC retry used `DVC_SITE_CACHE_DIR=/tmp/terser-dvc-site-cache` because the default `/var/tmp/dvc` is read-only. `core.autostage` was disabled in a temporary `.dvc/config.local` for each add; that file was removed afterward. No repository configuration remains changed. Full local verification output: [p2_smoke_local_checks.txt](p2_smoke_local_checks.txt).
