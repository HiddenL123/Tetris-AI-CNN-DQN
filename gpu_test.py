import torch

# Check if CUDA (GPU) is available
print(torch.version.cuda)        # Should print a CUDA version, e.g., '12.2'
print(torch.backends.cudnn.enabled)  # Should be True
print(torch.cuda.is_available())     # Should be True if GPU is accessible
# Check which device PyTorch will use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # Outputs 'cuda' or 'cpu'

# Optional: see GPU name
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))