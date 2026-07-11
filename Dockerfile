# ──────────────────────────────────────────────────────────────────────────────
# PanelCleaner – CPU image
# Build from a modern Python base, install core + server extras (no GPU).
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.12-slim

ARG USERNAME="pcleaner"
ARG USER_ID="1000"
ARG LOGIN_SHELL="/bin/bash"

RUN useradd -m -s "${LOGIN_SHELL}" -N -u "${USER_ID}" "${USERNAME}"

ENV USER=${USERNAME}
ENV UID=${USER_ID}
ENV HOME=/home/${USERNAME}
ENV PATH="${HOME}/.local/bin/:${PATH}"

# OpenCV and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        vim \
        nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PanelCleaner with CPU-only extras
COPY pyproject.toml setup.py ./
COPY pcleaner/ pcleaner/
COPY server/ server/
COPY main.py launch.py ./
RUN pip install --no-cache-dir -e ".[server,ocr,inpaint]"

RUN mkdir -p pcleaner/profiles temp_chapters credentials logs \
    && chown -R ${UID} /app

USER ${USERNAME}

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "server.api:app", "--host", "0.0.0.0", "--port", "8000"]
