import torch
from src.video_mode import gen_video
from src.common_constants import GenerationOptions as go

# Ensure PyTorch uses GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# Pass the device to gen_video function if needed
result = gen_video(input_video_path, output_path, generation_options, device=device)

print(result)
