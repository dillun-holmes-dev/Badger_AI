<h2 align="center">
  Manual to install TensorRT Containers in Lambda.ai instances 
</h2>

## Quick Start
### Lambda.ai
1. Go to [Lambda.ai](https://lambda.ai) and create an account.
2. Log in to your Lambda.ai account.
3. Click on the `Launch instance' button. It is located on the top right side of the website.
4. Select an instance. To replicate our results from the appendix, select `8x Tesla V100 (16 GB)`

### TensorRT Container Installation
1. Docker setup
   ```shell
   sudo usermod -aG docker $USER
   newgrp docker
   ```
3. Installing Nvidia DeepLearning container
   ```shell
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   ``` 
2. Installing TensorRT docker container
   ```shell
   docker pull nvcr.io/nvidia/tensorrt:24.04-py3
   docker run --gpus all -it --rm nvcr.io/nvidia/tensorrt:24.04-py3
   ```

3. Install the CUDA toolkit with the correct version (in our case 12.8)
    ```shell
    # cuda installation
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-8
    ```

The complete installation takes approximately 5 minutes.

## Installing DETRPose
### Quick Start
```shell
git clone https://github.com/SebastianJanampa/DETRPose.git
cd DETRPose
pip install -U pip
pip install -e ".[export,tensorrt]"
apt-get update && apt-get install libgl1
```

### Data Preparation
```
pip install gdown # to download files from google drive
gdown 1VprytECcLtU4tKP32SYi_7oDRbw7yUTL # images
unzip images.zip
```

### Usage
```shell
export model=l #n, s, m, l, x
mkdir trt_engines
```
1. Download official weights
    ```shell
    wget https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights/detrpose_hgnetv2_${model}.pth
    ```
2. Export onnx
    ```shell
    visionhub-export-onnx --check -c configs/detrpose/detrpose_hgnetv2_${model}_custom.py -r detrpose_hgnetv2_${model}.pth
    ```
3. Export tensorrt
    ```shell
    trtexec --onnx="onnx_engines/detrpose_hgnetv2_${model}.onnx" --saveEngine="trt_engines/detrpose_hgnetv2_${model}.engine" --fp16
    ```
4. Benchmark
    ```shell
    visionhub-benchmark-trt --infer_dir ./images --engine_dir trt_engines
    ```
