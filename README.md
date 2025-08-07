## 添加支持gpt-oss的vllm


```shell
uv pip install --pre vllm==0.10.1+gptoss     --extra-index-url https://wheels.vllm.ai/gpt-oss/     --extra-index-url https://download.pytorch.org/whl/nightly/cu128     --index-strategy unsafe-best-match --index-url https://mirrors.aliyun.com/pypi/simple
```