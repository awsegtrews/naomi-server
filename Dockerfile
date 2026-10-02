FROM python:3.12-slim
# запуск не від root (Hugging Face вимагає uid 1000, Render — байдуже)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR $HOME/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user *.py ./
COPY --chown=user static ./static
EXPOSE 7860
# Render передає порт у змінній PORT
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860} --proxy-headers --forwarded-allow-ips "*"
