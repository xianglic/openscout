ARG UBUNTU_VERSION=22.04
# This needs to generally match the container host's environment.
ARG CUDA_VERSION=11.7.1
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}

FROM ${BASE_CUDA_DEV_CONTAINER} as build

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=all

RUN apt-get update && \
    apt-get install -y build-essential git

WORKDIR /app

COPY . .

# Set nvcc architecture
ENV CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH}
# Enable cuBLAS
ENV LLAMA_CUBLAS=1

RUN make

FROM ${BASE_CUDA_RUN_CONTAINER} as runtime

COPY --from=build /app/main /main

# ENTRYPOINT [ "/main" ]

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install Flask
COPY server.py /app/server.py
# Flask uses the 5000 port by default, so we'll expose it.
EXPOSE 5000

# Instead of running main directly, we'll run the Flask server.
ENTRYPOINT [ "python3", "/app/server.py" ]
