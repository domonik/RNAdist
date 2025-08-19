FROM mambaorg/micromamba:1.5.6 as builder

# Install git only here
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Build the conda env and install pip package
COPY . /rnadist
WORKDIR /rnadist
RUN micromamba create -n myenv -f environment.yml && micromamba clean --all --yes && \
    micromamba run -n myenv pip install ./ --no-deps --no-build-isolation

RUN micromamba run -n myenv pip install pytest && \
    micromamba run -n myenv pytest --pyargs RNAdist



# -----------------------------
FROM mambaorg/micromamba:1.5.6 as runtime

COPY --from=builder /opt/conda/envs/myenv /opt/conda/envs/myenv
WORKDIR /app

ENV PATH=/opt/conda/envs/myenv/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/conda/envs/myenv/lib:$LD_LIBRARY_PATH

CMD ["/bin/bash"]
