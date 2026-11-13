FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    numpy==2.5.4 \
    pandas==3.1.1 \
    scikit-learn==1.9.1 \
    joblib==1.6.0

COPY pyproject.toml ./
COPY src/ src/
RUN pip install --no-cache-dir --no-deps .

COPY configs/ configs/

ENTRYPOINT ["python", "-m", "regressor"]
CMD ["--help"]
