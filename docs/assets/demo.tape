# Terminal demo for the README: Qwen3.8-Flash-Next (123.6 GiB) answering on one 32 GB RTX 5090.
# Needs a running server on :8090 (README "Quickstart: Qwen3.8-Flash-Next"), then from the repo root:
#   docker build -t imp-demo-vhs -f docs/assets/vhs.Dockerfile docs/assets
#   docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=utility --network host \
#     -v "$PWD/docs/assets":/vhs -v ~/models:/models:ro imp-demo-vhs demo.tape
Output demo.gif

Set Shell "bash"
Set FontSize 18
Set Width 1200
Set Height 640
Set Padding 20
Set Theme "Builtin Dark"
Set TypingSpeed 35ms
Set Framerate 20
Set PlaybackSpeed 1.0

Hide
Type "export PS1='$ ' && clear"
Enter
Show

Type "du -sh /models/Qwen3.8-Flash-Next-NVFP4"
Enter
Sleep 1.2s

Type "nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader"
Enter
Sleep 1.5s

Type `./demo-chat.sh "Write a Python function that merges overlapping intervals, with a short explanation."`
Enter
Sleep 5s
Sleep 2.5s
