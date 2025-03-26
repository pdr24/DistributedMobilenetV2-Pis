print(">>> TOP OF FILE REACHED")

'''
Running Instructions:

Master pi (rank 0): RANK=0 WORLD_SIZE=2 MASTER_ADDR=[master_pi_ip] MASTER_PORT=[port] python3 rpc_distributed_mobilenetv2.py

Pi 1: RANK=1 WORLD_SIZE=2 MASTER_ADDR=[master_pi_ip] MASTER_PORT=[port] python3 rpc_distributed_mobilenetv2.py

Note: single test image path is hard coded for now 
'''



import os
import torch
import torch.nn as nn
import torch.distributed.rpc as rpc
from torchvision import models, transforms
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

print(">>> Script has started parsing!")

# define global variables for RPC
global device, rank, world_size

def init_rpc():
    global device, rank, world_size

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rpc_backend_options = rpc.TensorPipeRpcBackendOptions(
        init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
    )

    rpc.init_rpc(
        name=f"worker{rank}",
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc_backend_options
    )


class Stage0(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(*list(model.features.children())[:7]).to(device)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x.to(device))


class Stage1(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.mobilenet_v2(pretrained=True)
        self.model = nn.Sequential(
            *list(model.features.children())[7:],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            *list(model.classifier.children())
        ).to(device)

    def forward(self, x):
        with torch.no_grad():
            return self.model(x.to(device))


def load_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor


stage1_rref = None  # Will be set by rank 1

def run_pipeline(image_tensor):
    print("[Rank 0] Running Stage 0")
    stage0 = Stage0()
    intermediate = stage0(image_tensor)

    print("[Rank 0] Sending tensor to Stage 1")
    output_fut = rpc.rpc_async("worker1", run_stage1, args=(intermediate.cpu(),))
    output = output_fut.wait()
    predicted_class = torch.argmax(output, dim=1).item()
    print(f"[Rank 0] Predicted class index: {predicted_class}")


def run_stage1(x):
    print("[Rank 1] Received tensor, running Stage 1")
    stage1 = Stage1()
    return stage1(x.to(device))


def main():
    print(f"[Rank {os.environ['RANK']}] Initializing RPC")
    init_rpc()

    if int(os.environ["RANK"]) == 0:
        image_path = "bird_imagenettest.jpeg"
        input_image = load_image(image_path)
        run_pipeline(input_image)

    rpc.shutdown()


if __name__ == "__main__":
    print(f"[Rank {os.environ.get('RANK', '?')}] Starting up on host: {os.uname()[1]}")
    main()
