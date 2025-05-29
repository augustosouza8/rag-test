from huggingface_hub import model_info

model_id = "deepseek-ai/DeepSeek-R1"
info = model_info(model_id)

total_size = 0

for file in info.siblings:
    if file.size is not None:
        size_mb = file.size / (1024 ** 2)
        total_size += size_mb
        print(f"{file.rfilename}: {size_mb:.2f} MB")
    else:
        print(f"{file.rfilename}: size unknown")

print(f"\nEstimated total size: {total_size:.2f} MB")
