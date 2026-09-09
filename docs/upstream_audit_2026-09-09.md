# Qwen3-TTS upstream audit — 2026-09-09

Authority is limited to the official Qwen repository, model card and PyPI project. Open pull
requests are experimental signals, not release contracts.

| Classification | Item | NeoVids/current upstream | Decision |
|---|---|---|---|
| RELEASED | `qwen-tts` | Production and latest PyPI are both `0.1.1` | Keep the pinned release. |
| RELEASED | 1.7B Base weights | Production and official model repository resolve revision `fd4b254389122332181a7c3db7f27e918eec64e3` | Keep the pinned revision. |
| MERGED MAIN | inference package | The installed `0.1.1` `qwen_tts` tree is byte-for-byte equivalent to official main for the inference package; later main history is not an inference release | Do not depend on git main. |
| OPEN PR | #362, `non_streaming_mode=True` default | Released signature defaults to `False`; the flag simulates non-streaming text input, not streaming audio generation | Keep `narration_v2`; candidate `narration_nonstreaming_v1` was benchmarked and rejected for production. |
| OPEN PR | #355, FP16 logits cast | Proposed float32 logits sampling is unmerged | No patch: repeated RTX 2060 FP16 runs reproduced no NaN, overflow or sampling exception. |
| OPEN PR | #360 Transformers 5 | Production/release remains on the upstream-pinned Transformers 4 line | No broad dependency upgrade. |
| OPEN PR | #191 `compile_codec` | Unmerged compile experiment | Do not add `torch.compile` to production. |
| OPEN PR | #234/#259 chunked-decode/sample-loss work | Unmerged and not required by the native 24 kHz physical-output path observed here | Watch only. |
| COMMUNITY EXPERIMENT | FlashAttention Turing forks, alternative runtimes | Not an official supported production path | Not adopted. RTX 2060 remains `float16` + `sdpa`, one GPU worker. |

Primary sources:

- https://pypi.org/project/qwen-tts/
- https://github.com/QwenLM/Qwen3-TTS
- https://github.com/QwenLM/Qwen3-TTS/blob/main/qwen_tts/inference/qwen3_tts_model.py
- https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- https://github.com/QwenLM/Qwen3-TTS/pull/362
- https://github.com/QwenLM/Qwen3-TTS/pull/355
- https://github.com/QwenLM/Qwen3-TTS/pull/360
- https://github.com/QwenLM/Qwen3-TTS/pull/191
- https://github.com/QwenLM/Qwen3-TTS/pull/234
- https://github.com/QwenLM/Qwen3-TTS/pull/259

## Local controlled evidence

All candidates used the same RTX 2060, text, frozen voice reference, hardware and seed `424242`.
Five-item PT-BR corpus categories were short, medium, long, GTA/Rockstar and long punctuation.

| Candidate | Speaker warnings | Content result | Timing/quality decision |
|---|---:|---|---|
| `narration_v2` | 0/5 | structural coverage passed; known GTA pronunciation disagreement retained | Production baseline. |
| `narration_nonstreaming_v1` | 2/5 | no general content gain | Not promoted; GTA similarity fell to `0.953306`. |
| `bernardo_icl_candidate_v1` | 1/5 | comparable structural coverage, altered pace | Experimental pending blind listening. |
| temp 0.80 / 0.70 | 0/1 | same GTA coverage as baseline | Insufficient corpus/listening evidence; not promoted. |

Sequential Portuguese → English → Portuguese generation reproduced the two Portuguese WAVs with
the same SHA and similarity, with stable latency and no growing cache/VRAM signal. Prompt-cache
telemetry showed four pinned entries, zero ephemeral entries and no eviction after the run.

The similarity calibration used the existing encoder, not another model. Six known-good samples
scored `0.990590–0.994125`; all 16 active development-block chunks scored
`0.987983–0.993230`. Suspect manual-job chunks included `0.968901` and `0.980853`, but its full
block scored `0.994105`, proving that block averaging can hide a wrong segment. Threshold `0.985`
is therefore a chunk-level warning only; human listening remains authoritative for borderline or
subjective cases.
