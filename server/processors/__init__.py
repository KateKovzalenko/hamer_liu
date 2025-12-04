import torch
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Running on CPU (Significant Latency Expected).")
else:
    print(f"Running on: {torch.cuda.get_device_name(0)}")