# Agnes 2.5 Flash Base

Official GitHub home for **Agnes 2.5 Flash Base**, an efficient sparse Mixture-of-Experts foundation model from Agnes AI.

> **Model weights and runtime support files are hosted on [Hugging Face](https://huggingface.co/Agnes-AI/Agnes-2.5-Flash-Base).** This repository is the official project landing page and deployment reference; it does not duplicate the approximately 190 GB FP8 checkpoint.

## Overview

Agnes 2.5 Flash Base is a decoder-only base model designed for long-context and high-throughput inference.

| Property | Value |
| --- | --- |
| Total parameters | 202B |
| Active parameters per token | Approximately 16B |
| Architecture | Sparse MoE with 160 routed experts, 6 selected per token, plus 1 shared expert |
| Precision | FP8 (E4M3) checkpoint with block-wise scaling |
| Context length | 1,048,576 tokens |
| License | Apache-2.0 |

This is a **base model** intended for continued pre-training, fine-tuning, and research. It has not undergone SFT or RLHF and should not be expected to reliably follow chat-style instructions without downstream adaptation.

## Get the Model

Download the official checkpoint, tokenizer, SGLang launcher, and version-pinned runtime patch from the model card:

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Agnes-AI/Agnes-2.5-Flash-Base \
  --local-dir ./Agnes-2.5-Flash-Base
```

## Serve with SGLang

The recommended serving path uses the stock `lmsysorg/sglang:v0.5.16` image. Agnes-specific SGLang support files and `serve.sh` are included in the Hugging Face model repository.

```bash
docker run --gpus all --shm-size 64g -p 30001:30002 \
  -v $(pwd)/Agnes-2.5-Flash-Base:/model \
  lmsysorg/sglang:v0.5.16 \
  bash /model/serve.sh
```

The launcher starts SGLang with tensor parallelism across 8 GPUs and exposes port `30002` inside the container. The image version must remain exactly `v0.5.16`; the supplied overlay targets that release only.

### Hardware Notes

- The FP8 checkpoint requires roughly 190 GB of GPU memory for weights alone.
- The default configuration uses 8-way tensor parallelism, for example 8 x H100 or H200 GPUs with at least 80 GB each.
- Model loading typically takes 10 to 15 minutes on the default configuration.

Check readiness after launch:

```bash
curl http://localhost:30001/health
curl http://localhost:30001/get_model_info
```

## Query the Server

Use the native generation endpoint:

```bash
curl http://localhost:30001/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The three laws of thermodynamics are",
    "sampling_params": {"max_new_tokens": 128, "temperature": 0.7, "top_p": 0.95}
  }'
```

For the OpenAI-compatible interface, use `/v1/completions`. As this is a base model, completions are preferred over chat completions.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:30001/v1", api_key="EMPTY")
response = client.completions.create(
    model="default",
    prompt="The three laws of thermodynamics are",
    max_tokens=128,
    temperature=0.7,
    top_p=0.95,
)
print(response.choices[0].text)
```

## Transformers

The model card provides `configuration_agnes.py` and `modeling_agnes.py` for use with Transformers. Enable remote code only after reviewing the source.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Agnes-AI/Agnes-2.5-Flash-Base"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="auto",
)
```

## Limitations and Responsible Use

- This release is not instruction-tuned or safety-aligned. Outputs can be incoherent, biased, or unsafe.
- Apply appropriate alignment, content filtering, evaluation, and operational safeguards before deployment.
- Single-GPU inference is not supported for the full FP8 checkpoint.

## Resources

- [Model weights, files, and full model card](https://huggingface.co/Agnes-AI/Agnes-2.5-Flash-Base)
- [Agnes AI](https://agnes-ai.com/)

## Citation

```bibtex
@misc{agnes2026flash,
  title={Agnes 2.5 Flash Base: An Efficient Sparse Mixture-of-Experts Foundation Model},
  author={Agnes AI Team},
  year={2026},
  url={https://huggingface.co/Agnes-AI/Agnes-2.5-Flash-Base}
}
```

## License

The code and model weights are released under the [Apache License 2.0](LICENSE).
