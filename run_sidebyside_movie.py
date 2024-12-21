import torch
from src.video_mode import gen_video
from src.common_constants import GenerationOptions as go

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set memory limit to 7.3GB
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(7.3 / torch.cuda.get_device_properties(0).total_memory)

input_video_path = input("Input video path: ")
output_path = "."

generation_options = {
    go.STEREO_DIVERGENCE.name.lower(): 2.0,
    go.STEREO_SEPARATION.name.lower(): 0.5,
    go.STEREO_MODES.name.lower(): ['left-right'],
    go.STEREO_BALANCE.name.lower(): 0.0,
    go.STEREO_OFFSET_EXPONENT.name.lower(): 1.0,
    go.STEREO_FILL_ALGO.name.lower(): 'polylines_sharp'
}

# Function to check GPU utilization
def check_gpu():
    if torch.cuda.is_available():
        print("GPU is available and being used.")
    else:
        print("GPU is not available. Using CPU.")

check_gpu()

# Pass the device to gen_video function
result = gen_video(input_video_path, output_path, generation_options, device=device)

print(result)
