# %%
# 
import json
import sys
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path

import einops
import numpy as np
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from rich import print as rprint
from rich.table import Table
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part2_cnns"
root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part2_cnns.tests as tests
import part2_cnns.utils as utils
from fundamentals_utils import CIFAR10
from plotly_utils import line

# On machines with many cores the default of one
# thread per core makes small CPU tensor operations much slower
t.set_num_threads(min(4, t.get_num_threads()))

# %%
# relu
class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.clip(x, min=0, max=None)


tests.test_relu(ReLU)

# %%
#
#  class torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
# Applies an affine linear transformation to the incoming data: y=xAT+b.
import math

class Linear(nn.Module):
    def __init__(self, num_in_features: int, num_out_features: int, bias=True):
        """
        A simple affine transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        """
        super().__init__()
        self.num_in_features = num_in_features
        self.num_out_features = num_out_features

        oversq = -1.0 / math.sqrt(num_in_features)

        rands = t.rand(size=(num_out_features, num_in_features))

        # shift from [0, 1) to between [-1, 1)
        weights = oversq * (2 * (rands) - 1)
        self.weight = nn.Parameter(weights)

        if bias:
            bias_tensor = oversq * (2 * t.rand(num_out_features) - 1)
            self.bias = nn.Parameter(bias_tensor)
        else:
            self.bias = None


    # Applies an affine linear transformation to the incoming data: y=xAT+b.
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (*, in_features)
        Return: shape (*, out_features)
        """

        result = x @ einops.rearrange(self.weight, "i j -> j i")
        if self.bias is not None:
            result += self.bias

        return result

    def extra_repr(self) -> str:
        return f"in_features={self.num_in_features}, out_features={self.num_out_features}, bias={self.bias is not None}"


tests.test_linear_parameters(Linear, bias=False)
tests.test_linear_parameters(Linear, bias=True)
tests.test_linear_forward(Linear, bias=False)
tests.test_linear_forward(Linear, bias=True)


# %%
# flatten
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: Tensor) -> Tensor:
        """
        Flatten out dimensions from start_dim to end_dim, inclusive of both.
        """
        shape = input.shape

        # Get start & end dims, handling negative indexing for end dim
        start_dim = self.start_dim
        end_dim = self.end_dim if self.end_dim >= 0 else len(shape) + self.end_dim

        # Get the shapes to the left / right of flattened dims, as well as size of flattened middle
        shape_left = shape[:start_dim]
        shape_right = shape[end_dim + 1 :]
        shape_middle = t.prod(t.tensor(shape[start_dim : end_dim + 1])).item()

        return t.reshape(input, shape_left + (shape_middle,) + shape_right)

    def extra_repr(self) -> str:
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["start_dim", "end_dim"]])
# %%
# MLP
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.relu = ReLU()
        self.linear1 = Linear(28*28, 100) # 28 by 28 pixels
        self.linear2 = Linear(100, 10) # 10 possible digits

    def forward(self, x: Tensor) -> Tensor:
        return self.linear2(self.relu(self.linear1.forward(self.flatten.forward((x)))))


tests.test_mlp_module(SimpleMLP)
tests.test_mlp_forward(SimpleMLP)

# %%
# training loop
MNIST_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Normalize([0.1307], [0.3081]),
    ]
)


def get_mnist(trainset_size: int = 10_000, testset_size: int = 1_000) -> tuple[Subset, Subset]:
    """Returns a subset of MNIST training data."""

    # Get original datasets, which are downloaded to "./data" for future use
    mnist_trainset = datasets.MNIST(exercises_dir / "data", train=True, download=True, transform=MNIST_TRANSFORM)
    mnist_testset = datasets.MNIST(exercises_dir / "data", train=False, download=True, transform=MNIST_TRANSFORM)

    # # Return a subset of the original datasets
    mnist_trainset = Subset(mnist_trainset, indices=range(trainset_size))
    mnist_testset = Subset(mnist_testset, indices=range(testset_size))

    return mnist_trainset, mnist_testset


mnist_trainset, mnist_testset = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=64, shuffle=True)
mnist_testloader = DataLoader(mnist_testset, batch_size=64, shuffle=False)

# Get the first batch of test data, by starting to iterate over `mnist_testloader`
for img_batch, label_batch in mnist_testloader:
    print(f"{img_batch.shape=}\n{label_batch.shape=}\n")
    break

# Get the first datapoint in the test set, by starting to iterate over `mnist_testset`
for img, label in mnist_testset:
    print(f"{img.shape=}\n{label=}\n")
    break

t.testing.assert_close(img, img_batch[0])
assert label == label_batch[0].item()

# %%
# training loop

device = t.device("mps" if t.backends.mps.is_available() else "cpu")
model = SimpleMLP().to(device)

batch_size = 128
epochs = 6

mnist_trainset, _ = get_mnist()
mnist_trainloader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)

optimizer = t.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []

for epoch in range(epochs):
    pbar = tqdm(mnist_trainloader)

    for imgs, labels in pbar:
        # Move data to device, perform forward pass
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)

        # Calculate loss, perform backward pass
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Update logs & progress bar
        loss_list.append(loss.item())
        pbar.set_postfix(epoch=f"{epoch + 1}/{epochs}", loss=f"{loss:.3f}", refresh=False)

line(
    loss_list,
    x_max=epochs * len(mnist_trainset),
    labels={"x": "Examples seen", "y": "Cross entropy loss"},
    title="SimpleMLP training on MNIST",
    width=700,
)

# %%
#

@dataclass
class SimpleMLPTrainingArgs:
    """
    Defining this class implicitly creates an __init__ method, which sets arguments as below, e.g.
    self.batch_size=64. Any of these fields can also be overridden when you create an instance, e.g.
    SimpleMLPTrainingArgs(batch_size=128).
    """

    batch_size: int = 64
    epochs: int = 6
    learning_rate: float = 1e-3


def train(args: SimpleMLPTrainingArgs) -> tuple[list[float], SimpleMLP]:
    """
    Trains & returns the model, using training parameters from the `args` object. Returns the model,
    and loss list.
    """
    model = SimpleMLP().to(device)

    mnist_trainset, _ = get_mnist()
    mnist_trainloader = DataLoader(mnist_trainset, batch_size=args.batch_size, shuffle=True)
    mnist_testloader = DataLoader(mnist_testset, batch_size=args.batch_size, shuffle=False)

    optimizer = t.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_list = []
    accuracy_list = []
    train_accuracy_list = []


    for epoch in range(args.epochs):
        pbar = tqdm(mnist_trainloader)
        num_correct_train = 0
        num_train_labels = 0
        for imgs, labels in pbar:
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)

            # Calculate loss, perform backward pass
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Update logs & progress bar
            loss_list.append(loss.item())
            pbar.set_postfix(epoch=f"{epoch + 1}/{args.epochs}", loss=f"{loss:.3f}", refresh=False)
            predictions = t.argmax(logits, dim=-1)
            num_correct_train += (predictions == labels).sum().item()
            num_train_labels += len(labels)

        train_accuracy_list.append(num_correct_train / num_train_labels)

        # measure accuracy here at end of epoch
        num_correct = 0
        num_labels = 0
        for imgs, labels in tqdm(mnist_testloader):
            # Move data to device, perform forward pass
            imgs, labels = imgs.to(device), labels.to(device)
            with t.inference_mode():
                logits = model(imgs)

            predictions = t.argmax(logits, dim=-1)
            num_correct += (predictions == labels).sum().item()
            num_labels += len(labels)

        accuracy = num_correct / num_labels
        accuracy_list.append(accuracy)

    return loss_list, accuracy_list, model, train_accuracy_list


args = SimpleMLPTrainingArgs()
loss_list, accuracy_list, model, train_accuracy_list = train(args)

line(
    y=[loss_list, [0.1] + accuracy_list],  # we start by assuming a uniform accuracy of 10%
    use_secondary_yaxis=True,
    x_max=args.epochs * len(mnist_trainset),
    labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="SimpleMLP training on MNIST",
    width=800,
)

line(
    y=[loss_list, [0.1] + train_accuracy_list],  # we start by assuming a uniform accuracy of 10%
    use_secondary_yaxis=True,
    x_max=args.epochs * len(mnist_trainset),
    labels={"x": "Num examples seen", "y1": "Cross entropy loss", "y2": "Train Accuracy"},
    title="SimpleMLP training on MNIST",
    width=800,
)


# %%
#
class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        """
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.

        We assume kernel is square, with height = width = `kernel_size`.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # YOUR CODE HERE - define & initialize `self.weight`
        kernel_height = kernel_size
        kernel_width = kernel_size
        weights = 2 * t.rand((out_channels, in_channels, kernel_height, kernel_width)) - 1
        k = 1 / math.sqrt(in_channels * kernel_height * kernel_width)
        self.weight = nn.Parameter(k * weights)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional conv2d, which you can import."""
        return t.nn.functional.conv2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_conv2d_module(Conv2d)
m = Conv2d(in_channels=24, out_channels=12, kernel_size=3, stride=2, padding=1)
print(f"Manually verify that this is an informative repr: {m}")

# %%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: int, stride: int | None = None, padding: int = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: Tensor) -> Tensor:
        """Call the functional version of maxpool2d."""
        return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        """Add additional information to the string representation of this class."""
        return ", ".join([f"{key}={getattr(self, key)}" for key in ["kernel_size", "stride", "padding"]])

# %%
class Sequential(nn.Module):
    _modules: dict[str, nn.Module]

    def __init__(self, *modules: nn.Module):
        super().__init__()
        for index, mod in enumerate(modules):
            self._modules[str(index)] = mod

    def __getitem__(self, index: int) -> nn.Module:
        index %= len(self._modules)  # deal with negative indices
        return self._modules[str(index)]

    def __setitem__(self, index: int, module: nn.Module) -> None:
        index %= len(self._modules)  # deal with negative indices
        self._modules[str(index)] = module

    def forward(self, x: Tensor) -> Tensor:
        """Chain each module together, with the output from one feeding into the next one."""
        for mod in self._modules.values():
            x = mod(x)
        return x

# %%

class BatchNorm2d(nn.Module):
    # The type hints below aren't functional, they're just for documentation
    running_mean: Float[Tensor, " num_features"]
    running_var: Float[Tensor, " num_features"]
    num_batches_tracked: Int[Tensor, ""]  # This is how we denote a scalar tensor

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        """
        Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.weight = nn.Parameter(t.ones(num_features))
        self.bias = nn.Parameter(t.zeros(num_features))

        self.register_buffer("running_mean", t.zeros(num_features))
        self.register_buffer("running_var", t.ones(num_features))
        self.register_buffer("num_batches_tracked", t.tensor(0))

    def forward(self, x: Float[Tensor, "batch channels height width"]) -> Float[Tensor, "batch channels height width"]:
        """
        Normalize each channel.

        In training mode, normalize using the biased variance `x.var(..., correction=0)`, but update
        `running_var` using the unbiased variance `x.var(..., correction=1)`.
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        """

        if self.training:
            # Use this data's mean & variance to normalize, then use it to update the buffers
            mean = x.mean(dim=(0, 2, 3))
            variance = x.var(dim=(0, 2, 3), correction=0)
            normalized_x = (x - mean.reshape(1, -1, 1, 1)) / (variance.reshape(1, -1, 1, 1) + self.eps).sqrt()
            with t.no_grad():
                variance_unbiased = x.var(dim=(0, 2, 3), correction=1)
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1 - self.momentum) * self.running_var + self.momentum * variance_unbiased
                self.num_batches_tracked += 1
        else:
            # Use the buffer mean & variance to normalize
            buffer_mean = self.running_mean.reshape(1, -1, 1, 1)
            buffer_variance = self.running_var.reshape(1, -1, 1, 1)
            normalized_x = (x - buffer_mean) / (buffer_variance + self.eps).sqrt()

        transformed_x = normalized_x * self.weight.T.reshape(1, -1, 1, 1) + self.bias.T.reshape(1, -1, 1, 1)
        return transformed_x

    def extra_repr(self) -> str:
        raise NotImplementedError()


tests.test_batchnorm2d_module(BatchNorm2d)
tests.test_batchnorm2d_forward(BatchNorm2d)
tests.test_batchnorm2d_running_mean(BatchNorm2d)
tests.test_batchnorm2d_running_stats_detached(BatchNorm2d)

# %%
class AveragePool(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        """

        return t.mean(x, dim=(2, 3))

tests.test_averagepool(AveragePool)

# %%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        """
        A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a
        `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right
        branch. Declare it second using another `Sequential`.
        """
        super().__init__()

        # Note - unless otherwise noted, you should assume convolutions have kernel_size=3, stride=1, padding=1 
        # (this is a shape preserving convolution i.e. the width & height of the input and output will be the same).
        # None of the convolutions have biases.
        is_shape_preserving = (first_stride == 1) and (in_feats == out_feats)  # determines if right branch is identity

        self.left = Sequential(
            Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1), # stridedconv
            BatchNorm2d(num_features=out_feats),
            ReLU(),
            Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(num_features=out_feats),
        )

        if is_shape_preserving:
            self.right = nn.Identity()
        else:
            self.right = Sequential(
                Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, padding=0), # stridedconv 
                BatchNorm2d(num_features=out_feats),
            )
        self.relu = ReLU()
  

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass. If no downsampling block is present, the addition should just add
        the left branch's output to the input.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / stride, width / stride)
        """

        result = self.left(x) + self.right(x)
        result = self.relu(result)

        return result


tests.test_residual_block(ResidualBlock)


# %%
#

class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        """
        An n_blocks-long sequence of ResidualBlock where only the first block uses the provided
        stride.
        """
        super().__init__()

        blocks = [ResidualBlock(in_feats=in_feats, out_feats=out_feats, first_stride=first_stride)]
        for _ in range(n_blocks - 1):
            blocks.append(ResidualBlock(out_feats, out_feats))

        self.sequence = Sequential(*blocks)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        """
        return self.sequence(x)


tests.test_block_group(BlockGroup)


# %%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        out_feats0 = 64
        self.n_blocks_per_group = n_blocks_per_group
        self.out_features_per_group = out_features_per_group
        self.first_strides_per_group = first_strides_per_group
        self.n_classes = n_classes

        self.in_layers = Sequential(
            Conv2d(3, out_feats0, kernel_size=7, stride=2, padding=3),
            BatchNorm2d(out_feats0),
            ReLU(),
            MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        residual_layers = []
        for i in range(len(n_blocks_per_group)):
            residual_layers.append(
                BlockGroup(
                    n_blocks=n_blocks_per_group[i],
                    in_feats=[64, *self.out_features_per_group][i],
                    out_feats=self.out_features_per_group[i],
                    first_stride=self.first_strides_per_group[i],
                )
            )
        self.residual_layers = Sequential(*residual_layers)

        self.out_layers = Sequential(
            AveragePool(),
            Linear(out_features_per_group[-1], n_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: shape (batch, channels, height, width)
        Return: shape (batch, n_classes)
        """
        post_first_conv_block = self.in_layers(x)
        post_block_groups = self.residual_layers(post_first_conv_block)
        logits = self.out_layers(post_block_groups)
        return logits



my_resnet = ResNet34()

# (1) Test via helper function `print_param_count`
target_resnet = models.resnet34()  # without supplying a `weights` argument, we just initialize with random weights
utils.print_param_count(my_resnet, target_resnet)

# (2) Test via `torchinfo.summary`
print("My model:", torchinfo.summary(my_resnet, input_size=(1, 3, 64, 64)), sep="\n")
print(
    "\nReference model:",
    torchinfo.summary(target_resnet, input_size=(1, 3, 64, 64), depth=2),
    sep="\n",
)

# %%

resnet = models.resnet34()
utils.print_param_count(resnet, resnet)
print(torchinfo.summary(resnet, input_size=(1, 3, 64, 64)))

# %%

def copy_weights(my_resnet: ResNet34, pretrained_resnet: models.resnet.ResNet) -> ResNet34:
    """Copy over the weights of `pretrained_resnet` to your resnet."""

    # Get the state dictionaries for each model, check they have the same number of parameters &
    # buffers
    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()
    assert len(mydict) == len(pretraineddict), "Mismatching state dictionaries."

    # Define a dictionary mapping the names of your parameters / buffers to their values in the
    # pretrained model
    state_dict_to_load = {
        mykey: pretrainedvalue
        for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items())
    }

    # Load in this dictionary to your model
    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet


pretrained_resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1).to(device)
my_resnet = copy_weights(my_resnet, pretrained_resnet).to(device)
print("Weights copied successfully!")

# %%
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]

IMAGE_FOLDER = section_dir / "resnet_inputs"

images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(t.float32, scale=True),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)

prepared_images = t.stack([IMAGENET_TRANSFORM(img) for img in images], dim=0).to(device)
assert prepared_images.shape == (len(images), 3, IMAGE_SIZE, IMAGE_SIZE)


# %%
@t.inference_mode()
def predict(
    model: nn.Module, images: Float[Tensor, "batch rgb h w"]
) -> tuple[Float[Tensor, " batch"], Int[Tensor, " batch"]]:
    """
    Returns the maximum probability and predicted class for each image, as a tensor of floats and
    ints respectively.
    """
    model.eval()
    logits = model(images)

    print(logits.shape)
    probabilities = logits.softmax(dim=-1)
    return probabilities.max(dim=-1)

with open(section_dir / "imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())

# Check your predictions match those of the pretrained model
my_probs, my_predictions = predict(my_resnet, prepared_images)
pretrained_probs, pretrained_predictions = predict(pretrained_resnet, prepared_images)
assert (my_predictions == pretrained_predictions).all()
t.testing.assert_close(my_probs, pretrained_probs, atol=5e-4, rtol=0)  # tolerance of 0.05%
print("All predictions match!")

# Print out your predictions, next to the corresponding images
for i, img in enumerate(images):
    table = Table("Model", "Prediction", "Probability")
    table.add_row("My ResNet", imagenet_labels[my_predictions[i]], f"{my_probs[i]:.3%}")
    table.add_row(
        "Reference Model",
        imagenet_labels[pretrained_predictions[i]],
        f"{pretrained_probs[i]:.3%}",
    )
    rprint(table)
    display(img)

# %%
# Images in [0, 1], without normalization
scale_to_unit_range = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(t.float32, scale=True),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
])
original_images: Float[Tensor, "batch rgb h w"] = t.stack([scale_to_unit_range(img) for img in images]).to(device)


class Normalized(nn.Module):
    """Applies ImageNet normalization inside the model so inputs live in [0, 1]."""
    def __init__(self, model: nn.Module, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", t.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", t.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        return self.model((x - self.mean) / self.std)


def fgsm(
    model: nn.Module,
    original_images: Float[Tensor, "batch rgb h w"],
    class_indices: Int[Tensor, " batch"],
    eps: float,
    targeted: bool = False,
) -> Float[Tensor, "batch rgb h w"]:
    """Perturb images with the Fast Gradient Sign Method (one step).

    Each element of the result differs from the input by at most `eps`. For an
    untargeted attack the step increases the cross-entropy loss for
    `class_indices`; for a targeted attack it decreases it.

    Args:
        model: Maps images in [0, 1] to logits of shape (batch, n_classes).
            Put into eval mode by this function.
        original_images: Clean images, shape (batch, rgb, h, w), values in [0, 1].
        class_indices: ImageNet class indices, shape (batch,). Classes to move
            away from if `targeted` is False, or toward if it is True.
        eps: Maximum per-element change (L-infinity bound), in the units of
            `original_images`.
        targeted: If True, decrease the loss for `class_indices` instead of
            increasing it.

    Returns:
        Adversarial images with the same shape and dtype as `original_images`,
        values clamped to [0, 1], detached from the autograd graph.
    """
    model.eval()
    images_requiring_grad: Float[Tensor, "batch rgb h w"] = original_images.clone().detach().requires_grad_(True)
    logits: Float[Tensor, "batch n_classes"] = model(images_requiring_grad) # raw output before softmax
    loss: Float[Tensor, ""] = F.cross_entropy(logits, class_indices)
    (grad,) = t.autograd.grad(loss, images_requiring_grad)
    noise: Float[Tensor, "batch rgb h w"]  = eps * grad.sign()
    adversarial_images: Float[Tensor, "batch rgb h w"] = original_images + (-noise if targeted else noise)
    return adversarial_images.clamp(0, 1).detach()


def tensor_to_pil(image: Float[Tensor, "rgb h w"]) -> Image.Image:
    """Convert a float image in [0, 1] to an 8-bit PIL image, rounding to nearest."""
    as_uint8 = (image.detach().clamp(0, 1) * 255).round().to(t.uint8).cpu()
    return TF.to_pil_image(as_uint8)


target_model = Normalized(my_resnet, IMAGENET_MEAN, IMAGENET_STD).to(device)
with t.no_grad():
    clean_predictions = target_model(original_images).argmax(-1)

eps = 4 / 255
noisy_images: Float[Tensor, "batch rgb h w"] = fgsm(target_model, original_images, clean_predictions, eps=eps)

with t.no_grad():
    adversarial_logits = target_model(noisy_images)
    adversarial_predictions = adversarial_logits.argmax(-1)
    adversarial_probabilities = adversarial_logits.softmax(-1).max(-1).values


delta = noisy_images - original_images
print(f"max |delta|: {delta.abs().max().item():.6f}  (eps = {eps:.6f})")
print(f"x_adversarial range: [{noisy_images.min().item():.3f}, {noisy_images.max().item():.3f}]")
assert delta.abs().max() <= eps + 1e-6
assert noisy_images.min() >= 0 and noisy_images.max() <= 1

# Clean vs adversarial predictions
with t.no_grad():
    clean_probs = target_model(original_images).softmax(-1).max(-1).values

from torchvision.transforms.v2 import functional as TF

for i in range(len(images)):
    table = Table("", "Prediction", "Probability")
    table.add_row("Clean", imagenet_labels[clean_predictions[i]], f"{clean_probs[i]:.3%}")
    table.add_row("FGSM", imagenet_labels[adversarial_predictions[i]], f"{adversarial_probabilities[i]:.3%}")
    rprint(table)
    display(tensor_to_pil(original_images[i]))   # clean, 224x224
    display(tensor_to_pil(noisy_images[i]))
    perturbation = noisy_images[i] - original_images[i]    # in [-eps, +eps]
    display(tensor_to_pil(perturbation / (2 * eps) + 0.5)) # 0.5 = no change  

success = (adversarial_predictions != clean_predictions).float().mean().item()
print(f"Attack success rate (prediction changed): {success:.0%}")
# %%
