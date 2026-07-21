FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

# --- Basic tools + link dependencies ---
RUN apt-get update && apt-get install -y \
    curl gnupg git sudo lsb-release ca-certificates \
    libopenblas-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- Install Bazelisk (as 'bazel') ---
RUN curl -L https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    -o /usr/local/bin/bazel \
    && chmod +x /usr/local/bin/bazel

WORKDIR /app

# --- Clone Drake and install its prerequisites ---
RUN git clone --depth 1 --branch v1.35.0 \
    https://github.com/RobotLocomotion/drake.git /app/drake
RUN yes | /app/drake/setup/ubuntu/install_prereqs.sh

# --- Install Gurobi 10.0 ---
COPY gurobi10.0.3_linux64.tar.gz /opt/
RUN cd /opt && tar xzf gurobi10.0.3_linux64.tar.gz && rm gurobi10.0.3_linux64.tar.gz

ENV GUROBI_HOME=/opt/gurobi1003/linux64
ENV PATH="${GUROBI_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${GUROBI_HOME}/lib"

# --- Copy just requirements first (better caching) ---
COPY requirements.txt /app/c3/requirements.txt
RUN pip3 install --no-cache-dir -r /app/c3/requirements.txt

# --- Copy your C3 project ---
COPY . /app/c3
WORKDIR /app/c3

# --- Build ---
RUN bazel build //...

CMD ["bash"]