from src.video_mode import gen_video
from src.common_constants import GenerationOptions as go

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 動画ファイルのパス
input_video_path = input("Input video path: ")

# 出力ディレクトリのパス
output_path = "."

# 生成オプション
generation_options = {
    go.STEREO_DIVERGENCE.name.lower(): 2.0,
    go.STEREO_SEPARATION.name.lower(): 0.5,
    go.STEREO_MODES.name.lower(): ['left-right'],
    go.STEREO_BALANCE.name.lower(): 0.0,
    go.STEREO_OFFSET_EXPONENT.name.lower(): 1.0,
    go.STEREO_FILL_ALGO.name.lower(): 'polylines_sharp'
}

# 動画からステレオイメージを生成
result = gen_video(input_video_path, output_path, generation_options, device=device)

print(result)
