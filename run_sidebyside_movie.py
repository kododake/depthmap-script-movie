import torch
import asyncio
from src.video_mode import gen_video
from src.common_constants import GenerationOptions as go
from config import MEMORY_LIMIT_GB  # Import from config.py

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Adjust memory limit dynamically
if device.type == "cuda":
    total_memory = torch.cuda.get_device_properties(0).total_memory
    memory_limit_gb = min(MEMORY_LIMIT_GB, total_memory / 10**9 * 0.8)  # Use 80% of total memory if it is lower
    torch.cuda.set_per_process_memory_fraction(memory_limit_gb * 10**9 / total_memory)

input_video_path = input("Input video path: ")
output_path = "outputs"  # Change output path to 'outputs' directory
custom_depthmap = input("Custom depthmap video path (press Enter to skip): ")

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

async def main():
    loop = asyncio.get_running_loop()

    # Handle case where custom depthmap is not provided
    global custom_depthmap  # Use global declaration
    if not custom_depthmap:
        custom_depthmap = None

    # Run gen_video in a separate thread to avoid blocking
    result = await loop.run_in_executor(None, gen_video, input_video_path, output_path, generation_options, custom_depthmap, device)

    # Empty cache to free up unused VRAM
    torch.cuda.empty_cache()
    
    print(result)

asyncio.run(main())
