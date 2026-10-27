# vhs plus curl for demo.tape (demo-chat.sh streams with curl). Build: docker build -t imp-demo-vhs -f vhs.Dockerfile .
FROM ghcr.io/charmbracelet/vhs:v0.12.1
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*
