FROM nvcr.io/nvidia/tensorrt:26.05-py3

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.11.9
ARG TENSORRT_PYTHON_VERSION=10.16.1.11

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/tensorrt/bin:/usr/local/bin:${PATH} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /tmp/python-build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    git \
    libbz2-dev \
    libffi-dev \
    libgdbm-dev \
    libgl1 \
    libglib2.0-0 \
    liblzma-dev \
    libncursesw5-dev \
    libnss3-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    tk-dev \
    uuid-dev \
    wget \
    xz-utils \
    zlib1g-dev \
    && wget -O Python.tgz "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
    && tar -xzf Python.tgz \
    && cd "Python-${PYTHON_VERSION}" \
    && ./configure --enable-shared --with-ensurepip=install \
    && make -j"$(nproc)" \
    && make altinstall \
    && ln -sf /usr/local/bin/python3.11 /usr/local/bin/python \
    && ln -sf /usr/local/bin/pip3.11 /usr/local/bin/pip \
    && printf '/usr/local/lib\n' > /etc/ld.so.conf.d/python3.11.conf \
    && ldconfig \
    && python --version \
    && pip install --upgrade pip setuptools wheel \
    && rm -rf /var/lib/apt/lists/* /tmp/python-build

COPY docker_requirements.txt /tmp/docker_requirements.txt

RUN pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1 \
    && pip install -r /tmp/docker_requirements.txt \
    && pip install "tensorrt-cu13==${TENSORRT_PYTHON_VERSION}"

WORKDIR /workspace/VisionHub

COPY . /workspace/VisionHub

RUN pip install --no-cache-dir -e ".[export, tensorrt]" 

CMD ["bash"]
