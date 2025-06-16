# HUAWEI-glasses
边侧模型：

# 1. 语音转文本 sherpa-onnx (sherpa-onnx) 10003

```shell
source /data/zdg/software/activate-cuda-11.4.sh
conda activate sherpa-onnx
cd /data/zdg/sherpa-onnx/
CUDA_VISIBLE_DEVICES=1 python streaming_server_with_sid.py
```



# 2.  文本转语音xtts (coqui-114) 10009 10010

```shell
 conda activate coqui
 cd /data/zdg/coqui-ai-TTS
 #python tts_api.py
 CUDA_VISIBLE_DEVICES=1 python tts_api_gender.py
```



# 3. 性别识别（deepface）10004

```shell
 conda activate deepface
 cd /data/zdg/deepface/
 CUDA_VISIBLE_DEVICES=python gender_api.py
```



# 4. qwen 

```shell
cd /data/jxq/edge_compute/llama.cpp/build-cuda/bin
CUDA_VISIBLE_DEVICES=1 ./llama-server --model ../../../qwen25/qwen2.5-7b-instruct-fp16.gguf --port 8080 --flash-attn
```

```shell
llama.cpp
合并gguf模型
./llama-gguf-split --merge ../../../qwen25/qwen2.5-7b-instruct-fp16-00001-of-00004.gguf ../../../qwen25/qwen2.5-7b-instruct-fp16.ggu

serve模型
/data/jxq/edge_compute/llama.cpp/build-cuda/bin
CUDA_VISIBLE_DEVICES="-0" ./llama-server --model ../../../qwen25/qwen2.5-7b-instruct-fp16.gguf --port 8080 --flash-attn enabled

参数
https://github.com/ggml-org/llama.cpp/blob/be7c3034108473beda214fd1d7c98fd6a7a3bdf5/examples/server/README.md

pheonix项目地址：/data/jxq/edge_compute/
llama_server: /data/jxq/edge_compute/llama.cpp/build-cuda/bin/llama-server
```

# 5. 8765

```shell
cd /data/zdg/ws_server
conda activate openai
python web_socket_server_ai.py
```

