# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    make \
    pandoc \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Quarto
RUN curl -LO https://github.com/quarto-dev/quarto-cli/releases/download/v1.4.550/quarto-1.4.550-linux-amd64.deb \
    && dpkg -i quarto-1.4.550-linux-amd64.deb \
    && rm quarto-1.4.550-linux-amd64.deb

# Install TinyTeX (for PDF generation in Quarto)
RUN quarto install tinytex --no-prompt

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit and Jupyter ports
EXPOSE 8501
EXPOSE 8888

# Default command: Run the streamlit app
CMD ["streamlit", "run", "exam_project/src/app.py"]
