import os
import torch
from torch import nn
import argparse
import data_setup, engine, model_builder, utils

from torchvision import transforms

# 创建参数解析器
parser = argparse.ArgumentParser(description="Train a TinyVGG model")

# 添加超参数和配置参数
parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (default: 5)")
parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
parser.add_argument("--hidden-units", type=int, default=10, help="Number of hidden units (default: 10)")
parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate (default: 0.001)")
parser.add_argument("--train-dir", type=str, default="data/pizza_steak_sushi/train", help="Training data directory")
parser.add_argument("--test-dir", type=str, default="data/pizza_steak_sushi/test", help="Testing data directory")
parser.add_argument("--model-save-dir", type=str, default="models", help="Directory to save models")
parser.add_argument("--model-name", type=str, default="05_going_modular_script_mode_tinyvgg_model.pth", help="Model file name")

# 解析参数
args = parser.parse_args()

# Deleted:NUM_EPOCHS = 5
# Deleted:BATCH_SIZE = 32
# Deleted:HIDDEN_UNITS = 10
# Deleted:LEARNING_RATE = 0.001
# Deleted:train_dir = "data/pizza_steak_sushi/train"
# Deleted:test_dir = "data/pizza_steak_sushi/test"

NUM_EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
HIDDEN_UNITS = args.hidden_units
LEARNING_RATE = args.learning_rate
train_dir = args.train_dir
test_dir = args.test_dir

device = "cuda" if torch.cuda.is_available() else "cpu"

data_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform = data_transform,
    batch_size = BATCH_SIZE
)

model = model_builder.TinyVGG(
    input_shape = 3,
    hidden_shape = HIDDEN_UNITS,
    output_shape = len(class_names)
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

utils.save_model(model=model,
                 target_dir=args.model_save_dir,
                 model_name=args.model_name)
