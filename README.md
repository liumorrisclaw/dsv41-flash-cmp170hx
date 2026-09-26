# DeepSeek V4.1-Flash on 4× CMP 170HX — 62GB RAM, SATA, zero new hardware

**Global first**: Full 764B DeepSeek V4.1-Flash (with 189GB Engram retrieval tables, vision, DSpark speculation) running on 4× CMP 170HX mining cards with **62GB host RAM and a SATA SSD** — no added RAM, no NVMe, no hardware changes.

The trick: treat the 189GB Engram n-gram retrieval table as a **database, not a weight**. Keep it on disk, query it on demand via `pread` through the Linux page cache (~40GB of RAM acting as LRU), and never load it into memory.

中文故事版: [docs/dsv41-cmp170-x-article-zh-external.md](docs/dsv41-cmp170-x-article-zh-external.md)

## Results

| Metric | This build | kaka86mm (373GB RAM + NVMe) | Mia 2× DGX Spark (¥70k) |
|---|---|---|---|
| prefill 16K | **3,355** tok/s | 2,119 | 987 |
| prefill 32K steady | **1,821** | 1,453 | 1,055 |
| decode counting | **59.9** | 66.7 | 40 |
| decode code | **35.5** | 44.3 | 43.0 |
| context | **1M tested** | 512K | 600K |
| quality | 5/7 + vision 2/2 | 7/7 + vision 2/2 | — |
| host RAM | **62GB** | 373GB | ~128GB |
| storage | **SATA** | NVMe | NVMe |
| cost | ~¥50k | ~¥60k | ~¥70k |

## Quick start

```bash
# Prerequisites: cmpunlocker patched driver (10de:20c2), 4× CMP 170HX, vLLM image
docker pull vllm/vllm-openai:deepseekv41-flash-0909

# 1. Download weights (334GB, contains EXL3 2.0bpw + FP8 Engram)
#    sfxnz/DeepSeek-V4.1-Flash-EXL3 @ branch 2.0bpw-mcg

# 2. Clone overlay + patches
git clone https://github.com/zebgop-ops/dsv41reap-pp.git ~/v41-graft/dsv41reap
git clone https://github.com/vcruz305/vllm-exl3.git ~/v41-graft/vllm-exl3
git clone https://github.com/turboderp-org/exllamav3.git ~/v41-graft/exllamav3
cd ~/v41-graft/exllamav3 && git checkout v1.5.0

# 3. Seed + patch overlay (26 patches, syntax-checked)
cd ~/v41-graft/dsv41reap/overlay
DSV41_IMG=vllm/vllm-openai:deepseekv41-flash-0909 bash make-overlay.sh
bash apply-overlay.sh

# 4. Build native components
docker run --rm -v $PWD/hybrid:/w --entrypoint bash vllm/vllm-openai:deepseekv41-flash-0909 /w/build.sh
docker run --rm -v $PWD/engram_ssd:/w --entrypoint bash vllm/vllm-openai:deepseekv41-flash-0909 \
  -c "cd /w && g++ -O2 -std=c++17 -shared -fPIC row_store.cpp -o librow_store.so -lpthread"

# 5. Build runtime image (vllm-exl3 + exllamav3 sm80)
#    See scripts/ and docs/deployment-plan.md for full details

# 6. Launch with SSD-Engram mode
bash serve-dsv41-ssd.sh   # reads from /models/DSV41-2bpw, serves on :8095
```

**Critical environment variables:**
```bash
DSV41_ENGRAM_STORAGE=ssd          # Engram stays on disk, page cache as LRU
DSV41_SKIP_WEIGHT_RE='layers\.(1|14)\.engram\.embed\.'  # skip 189GB at load time
vm.overcommit_memory=1            # required: 62GB RAM can't mmap 94.5GB otherwise
```

## Repo contents

| Path | Description |
|---|---|
| `docs/` | Deployment plan, performance comparison, Engram optimization, X article |
| `results/` | All benchmark JSONs (ladder, dimension matrix, 1M, disk IO sampling) |
| `scripts/` | Launch script (SSD-Engram mode), prewarm, benchmark tools, rollback |

## The SSD-Engram insight

The Engram isn't a computation weight — it's an n-gram hash lookup table. You query it (~24 rows × 264 bytes per token), you don't matrix-multiply it. So treat it like a database:

- **Never load it into memory.** Skip it at weight-load time with a regex.
- **Query it on demand** via `pread` through the page cache, inside a CUDA host callback (legal inside CUDA graphs).
- **Let the kernel manage caching.** ~40GB of RAM as page cache covers the hot zone via Zipf distribution.
- **Accept the prefill penalty.** Cold prefill does real disk I/O (31-40KB/token). Hot prefill (prefix cache hit) reads zero from disk.

Disk lifetime impact: **zero** (read-only, no NAND wear).

## Known limitations

- **mcg quantization pack** has repetition stickiness on greedy prose — mitigated by `generation_config.json` (temp 0.3, freq_penalty 0.6, rep_penalty 1.1). Pollard-calibrated pack (kaka86mm) would give 7/7 quality.
- **Concurrent prefill doesn't scale** — SATA random-read bandwidth (~54MB/s) is the ceiling. NVMe would fix this.
- **Cold start** first long document is slow (~870 tok/s @ 32K) until page cache warms up.

## Credits

- [zebgop-ops/dsv41reap-pp](https://github.com/zebgop-ops/dsv41reap-pp) — SSD-Engram mechanism, full vLLM overlay
- [kaka86mm/dsv41-flash-pp4-170hx](https://github.com/kaka86mm/dsv41-flash-pp4-170hx) — EXL3 pack, benchmark suite
- [amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) — CMP 170HX unlock toolchain
- [0xSero](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) — row_store.cpp prototype
- [Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) — knowledge base
- [sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3/tree/2.0bpw-mcg) — weights

## License

MIT
