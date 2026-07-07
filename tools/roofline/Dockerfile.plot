# Plot container for the roofline pipeline (host stays clean — no matplotlib
# on the host). Built on demand by `roofline plot`.
FROM python:3.14-slim
RUN pip install --no-cache-dir matplotlib==3.9.2
