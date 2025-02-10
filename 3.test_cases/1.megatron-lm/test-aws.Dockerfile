# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

# DOCKER_BUILDKIT=1 docker build --progress plain -t aws-nemo-megatron:latest .

FROM nvcr.io/nvidia/pytorch:24.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV EFA_INSTALLER_VERSION=latest
ENV AWS_OFI_NCCL_VERSION=v1.13.2-aws
ENV NCCL_TESTS_VERSION=master

RUN apt-get update -y
RUN apt-get remove -y --allow-change-held-packages \
                      libmlx5-1 ibverbs-utils libibverbs-dev libibverbs1

RUN rm -rf /opt/hpcx/ompi \
    && rm -rf /usr/local/mpi \
    && rm -fr /opt/hpcx/nccl_rdma_sharp_plugin \
    && ldconfig
ENV OPAL_PREFIX=
RUN apt-get install -y --allow-unauthenticated \
    git \
    gcc \
    vim \
    kmod \
    openssh-client \
    openssh-server \
    build-essential \
    curl \
    autoconf \
    libtool \
    gdb \
    automake \
    cmake \
    apt-utils \
    libhwloc-dev \
    aptitude && \
    apt autoremove -y

# Uncomment below stanza to install the latest NCCL
# Require efa-installer>=1.29.0 for nccl-2.19.0 to avoid libfabric gave NCCL error.
ENV NCCL_VERSION=v2.24.3-1
RUN apt-get remove -y libnccl2 libnccl-dev \
   && cd /tmp \
   && git clone https://github.com/NVIDIA/nccl.git -b ${NCCL_VERSION} \
   && cd nccl \
   && make -j src.build BUILDDIR=/usr/local \
   # nvcc to target p5 and p4 instances
   NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_80,code=sm_80" \
   && rm -rf /tmp/nccl

# EFA
RUN apt-get update && \
    apt-get install -y libhwloc-dev && \
    cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz  && \
    tar -xf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    ./efa_installer.sh -y -g -d --skip-kmod --skip-limit-conf --no-verify && \
    ldconfig && \
    rm -rf /tmp/aws-efa-installer /var/lib/apt/lists/*

## Install AWS-OFI-NCCL plugin
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y libhwloc-dev
#Switch from sh to bash to allow parameter expansion
SHELL ["/bin/bash", "-c"]
RUN curl -OL https://github.com/aws/aws-ofi-nccl/releases/download/${AWS_OFI_NCCL_VERSION}/aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && tar -xf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz \
    && cd aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && ./configure --prefix=/opt/aws-ofi-nccl/install \
        --with-mpi=/opt/amazon/openmpi \
        --with-libfabric=/opt/amazon/efa \
        --with-cuda=/usr/local/cuda \
        --enable-platform-aws \
    && make -j $(nproc) \
    && make install \
    && cd .. \
    && rm -rf aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v} \
    && rm aws-ofi-nccl-${AWS_OFI_NCCL_VERSION//v}.tar.gz

SHELL ["/bin/sh", "-c"]

# NCCL
RUN echo "/usr/local/lib"      >> /etc/ld.so.conf.d/local.conf && \
    echo "/opt/amazon/openmpi/lib" >> /etc/ld.so.conf.d/efa.conf && \
    ldconfig

ENV OPAL_PREFIX=/opt/amazon/openmpi \
    NCCL_SOCKET_IFNAME=^docker,lo 
#    OMPI_MCA_pml=^cm,ucx            \
#    OMPI_MCA_btl=tcp,self           \
#    OMPI_MCA_btl_tcp_if_exclude=lo,docker0 \

# NCCL-tests
RUN git clone https://github.com/NVIDIA/nccl-tests.git /opt/nccl-tests \
    && cd /opt/nccl-tests \
    && git checkout ${NCCL_TESTS_VERSION} \
    && make MPI=1 \
    MPI_HOME=/opt/amazon/openmpi \
    CUDA_HOME=/usr/local/cuda \
    # nvcc to target p5 and p4 instances
    NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_80,code=sm_80"

# Custom libraries version
WORKDIR /workspace/

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

## Transformer Engine
ENV NVTE_WITH_USERBUFFERS=1
ARG TE_REVISION=v1.13
ENV CUSTOM_TE_REVISION ${TE_REVISION}
RUN pip install wandb && \
    pip install transformers && \
    pip uninstall -y transformer-engine
RUN if [ "${TE_REVISION}" != SKIP ]; then \
      NVTE_UB_WITH_MPI=1 MPI_HOME=/opt/amazon/openmpi pip install --force-reinstall --no-deps git+https://github.com/NVIDIA/TransformerEngine.git@${TE_REVISION} \
    ; fi


# Fix HF and pkg_resources import issues (remove after it is fixed)
RUN pip install -U huggingface_hub
RUN pip install setuptools==69.5.1
RUN pip install pytorch-lightning==2.4.0

## fix opencc
RUN apt-get update && apt-get install -y --no-install-recommends libopencc-dev

# Benchmark code
WORKDIR /workspace

ENV PYTHONPATH "/workspace:${PYTHONPATH}"
