<h2 align="center">
  Manual to install TensorRT in Lambda.ai instances 
</h2>

## Quick Start
### Lambda.ai
1. Go to [Lambda.ai](https://lambda.ai) and create an account.
2. Log in to your Lambda.ai account.
3. Click on the `Launch instance' button. It is located on the top right side of the website.
4. Select an instance. To replicate our results from the appendix, select `1x A10 (24 GB PCle)`

### CUDA Installation
The Lambda Stack installs a pre-packaged version of CUDA with only whats needed for typical deep learning workflows. 
But the `.deb` TensorRT installation expects the full CUDA Toolkit to already be installed in the system in the standard way via NVIDIAs `.deb` repo. 
Thats why your TensorRT installation only succeeded after installing CUDA. 
This ensured all the expected binaries, libraries, and metadata were in place for TensorRT to install cleanly.

1. Check which CUDA version your Lambda.ai instance is using
    ```shell
    nvidia-smi
    ```
    We got the following output
    ```shell
    +-----------------------------------------------------------------------------------------+
    | NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
    |-----------------------------------------+------------------------+----------------------+
    | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
    |                                         |                        |               MIG M. |
    |=========================================+========================+======================|
    |   0  NVIDIA A10                     On  |   00000000:06:00.0 Off |                    0 |
    |  0%   28C    P8              9W /  150W |       1MiB /  23028MiB |      0%      Default |
    |                                         |                        |                  N/A |
    +-----------------------------------------+------------------------+----------------------+
                                                                                             
    +-----------------------------------------------------------------------------------------+
    | Processes:                                                                              |
    |  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
    |        ID   ID                                                               Usage      |
    |=========================================================================================|
    |  No running processes found                                                             |
    +-----------------------------------------------------------------------------------------+
    ```

2. Install the CUDA toolkit with the correct version (in our case 12.8)
    ```shell
    # cuda installation
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-12-8
    ```

3. Install TensorRT

    When you use the `.deb` installation, you will install the latest TensorRT.
    ```shell
    #tensorrt installation
    wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.9.0/local_repo/nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-12.8_1.0-1_amd64.deb
    sudo dpkg -i nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-12.8_1.0-1_amd64.deb 
    sudo cp /var/nv-tensorrt-local-repo-ubuntu2204-10.9.0-cuda-12.8/nv-tensorrt-local-AD7406A2-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo apt-get install tensorrt
    ```

The complete installation takes approximately 10-15 minutes.

## Installing DETRPose
### Quick Start
```shell
git clone https://github.com/SebastianJanampa/DETRPose.git
cd DETRPose
pip install -U pip
pip install -e ".[export,tensorrt]"
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
    alias trtexec="/usr/src/tensorrt/bin/trtexec"
    trtexec --onnx="onnx_engines/detrpose_hgnetv2_${model}.onnx" --saveEngine="trt_engines/detrpose_hgnetv2_${model}.engine" --fp16
    ```
4. Benchmark
    ```shell
    visionhub-benchmark-trt --infer_dir ./images --engine_dir trt_engines
    ```
