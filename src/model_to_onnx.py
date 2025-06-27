# This script is not ment tu be run on jetson nano 
import torch

model = torch.load('sea-form8_sea_aug-random_best.pt', map_location='cpu', weights_only=False)

model.eval()

dummy_input = torch.randn(1, 3, 800, 608).half()

torch.onnx.export(
        model["model"],                      # PyTorch model
        dummy_input,                # Example input
        "model_2.onnx",               # Output file
        opset_version=11,           # ONNX opset version
        input_names=['input'],      # Input name
        output_names=['output'],     # Output name
        dynamic_axes={              # Dynamic axes if needed
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True                # Show export details
    )
