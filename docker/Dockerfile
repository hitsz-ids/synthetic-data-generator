FROM python:3.10.12-slim-bookworm as builder

RUN pip install build twine hatch
COPY . /source
WORKDIR /source
RUN python -m build

FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

COPY --from=builder /source/dist/*.whl /tmp/

RUN pip install --no-cache-dir /tmp/*.whl
