FROM python:3.11-slim

WORKDIR /app

# Install openenv-core first without deps to avoid gradio backtracking,
# then install actual runtime deps from requirements.txt.
RUN pip install --no-cache-dir --no-deps openenv-core>=0.2.3 fastmcp>=3.0.0
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# HF Spaces expects port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
