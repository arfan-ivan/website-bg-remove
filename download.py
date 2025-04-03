import os
import urllib.request

def download_u2net_model(model_url, save_path):
    """Download U-2-Net model from the given URL."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    model_file = os.path.join(save_path, 'u2net.pth')

    try:
        print(f"Downloading U-2-Net model from {model_url}...")
        urllib.request.urlretrieve(model_url, model_file)
        print(f"Model downloaded successfully and saved at {model_file}")
    except Exception as e:
        print(f"An error occurred: {e}")

# URL untuk mengunduh model U-2-Net
model_url = "https://github.com/xuebinqin/U-2-Net/releases/download/v1.0/u2net.pth"
# Direktori tempat menyimpan model
save_path = "models/u2net"

# Panggil fungsi untuk mendownload model
download_u2net_model(model_url, save_path)
