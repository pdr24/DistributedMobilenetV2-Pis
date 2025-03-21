import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torchvision import models, transforms
from PIL import Image
from torch.distributed.pipelining import ScheduleGPipe, PipelineStage
from dotenv import load_dotenv


load_dotenv()


print(">>> Script has started parsing!") # for testing purposes 

# define global variables for pipeline setup
global rank, device, pp_group, stage_index, num_stages

def init_distributed():
    global rank, device, pp_group, stage_index, num_stages

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    stage_index = rank
    num_stages = world_size

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize default process group
    dist.init_process_group(backend='gloo')

    # create pipeline parallelism group (just the default group for now)
    pp_group = dist.new_group()

def split_mobilenet_v2():
    """
    Split MobileNetV2 into two stages for model parallelism.
    - Stage 0 --> Features[:7]
    - Stage 1 --> Features[7:] & Classifier
    """
    model = models.mobilenet_v2(pretrained=True)

    if stage_index == 0:
        # stage 0: first half of features
        stage_model = nn.Sequential(*list(model.features.children())[:7])
    elif stage_index == 1:
        # stage 1: second half + classifier
        stage_model = nn.Sequential(
            *list(model.features.children())[7:],
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            *list(model.classifier.children())
        )

    return PipelineStage(
        stage_model.to(device),
        stage_index,
        num_stages,
        device
    )

def load_image(image_path):
    """
    Load and preprocess image for MobileNetV2.
    Returns a tensor shaped (1, 3, 224, 224)
    """
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

def main():
    print("entering main()")
    init_distributed()

    # load test image (from NFS)
    print("loading test image")
    test_image_path = "/home/cc/datasets/mobilenet_test/bird_imagenettest.jpeg"  # change path as needed
    input_image = load_image(test_image_path)
    input_image = input_image.to(device)

    # split model
    stage = split_mobilenet_v2()

    # no training, so dummy loss
    def dummy_loss_fn(output, target):
        return torch.tensor(0.0, requires_grad=True)

    # setup ScheduleGPipe
    print("setting up ScheduleGPipe")
    schedule = ScheduleGPipe(stage, n_microbatches=1, loss_fn=dummy_loss_fn)

    if rank == 0:
        # stage 0: send input
        print("we're in rank 0")
        schedule.step(input_image)
    elif rank == 1:
        # stage 1: receive final output
        print("we're in rank 1")
        outputs = schedule.step()
        predicted_class = torch.argmax(outputs[0], dim=1).item()
        print(f"Predicted class index: {predicted_class}")

    dist.destroy_process_group()

if __name__ == "__main__":
    print(f"[Rank {os.environ.get('RANK', '?')}] Starting up on host: {os.uname()[1]}")
    main()

