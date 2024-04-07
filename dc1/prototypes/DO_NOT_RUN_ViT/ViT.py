from torchvision.models.vision_transformer import VisionTransformer, ConvStemConfig, Encoder, EncoderBlock
from typing import Any, Callable, Dict, List, NamedTuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt  # type: ignore
from functools import partial
from collections import OrderedDict
import math

from dc1.image_dataset import ImageDataset, Path


class ModifiedViT(nn.Module):
    """
    Modified version of the Visual Transformer Class.
    """

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0.0,
            attention_dropout: float = 0.0,
            num_classes: int = 1000,
            representation_size: Optional[int] = None,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        torch._assert(image_size % patch_size == 0, "Input shape indivisible by patch size!")
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.representation_size = representation_size
        self.norm_layer = norm_layer

        self.conv_proj = nn.Conv2d(
            in_channels=1, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
        )

        seq_length = (image_size // patch_size) ** 2

        # Add a class token
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        seq_length += 1

        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer,
        )
        self.seq_length = seq_length
        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if representation_size is None:
            heads_layers["head"] = nn.Linear(hidden_dim, num_classes)
        else:
            heads_layers["pre_logits"] = nn.Linear(hidden_dim, representation_size)
            heads_layers["act"] = nn.Tanh()
            heads_layers["head"] = nn.Linear(representation_size, num_classes)

        self.heads = nn.Sequential(heads_layers)

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))
            if self.conv_proj.bias is not None:
                nn.init.zeros_(self.conv_proj.bias)
        elif self.conv_proj.conv_last is not None and isinstance(self.conv_proj.conv_last, nn.Conv2d):
            # Init the last 1x1 conv of the conv stem
            nn.init.normal_(
                self.conv_proj.conv_last.weight, mean=0.0, std=math.sqrt(2.0 / self.conv_proj.conv_last.out_channels)
            )
            if self.conv_proj.conv_last.bias is not None:
                nn.init.zeros_(self.conv_proj.conv_last.bias)
        if hasattr(self.heads, "pre_logits") and isinstance(self.heads.pre_logits, nn.Linear):
            fan_in = self.heads.pre_logits.in_features
            nn.init.trunc_normal_(self.heads.pre_logits.weight, std=math.sqrt(1 / fan_in))
            nn.init.zeros_(self.heads.pre_logits.bias)

        if isinstance(self.heads.head, nn.Linear):
            nn.init.zeros_(self.heads.head.weight)
            nn.init.zeros_(self.heads.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        p = self.patch_size
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x: torch.Tensor):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)
        print(x.shape)
        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


class FullImageEmbedder(nn.Module):
    """
    Encoder or the whole image, as per https://www.sciencedirect.com/science/article/pii/S0169260722005223
    """

    def __init__(self, D, l1_filters=16, l1_kernel=3, l1_stride=1, l2_filters=256, l2_kernel=5, l2_stride=1,
                 l3_kernel=5, l3_stride=1):
        super().__init__()
        self.D = D  # D is the dimension of the vector
        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(1, l1_filters, kernel_size=l1_kernel, stride=l1_stride),
            nn.Conv2d(l1_filters, l2_filters, kernel_size=l2_kernel, stride=l2_stride),
            nn.Conv2d(l2_filters, D, kernel_size=l3_kernel, stride=l3_stride))

    def forward(self, x: torch.Tensor):
        x = self.cnn_layers(x)
        # global max pooling
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        shape = x.shape
        # print(f"Initial embedded x shape: {shape}")
        x = x.view(shape[0], shape[2], shape[1])
        return x


class IEncoder(nn.Module):
    """
    Encoder used for IEViT.
    """

    def __init__(
            self,
            seq_length: int,
            num_layers: int,
            num_heads: int,
            hidden_dim: int,
            mlp_dim: int,
            dropout: float = 0,
            attention_dropout: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_layers = num_layers

        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))  # from BERT
        self.dropout = nn.Dropout(dropout)

        layers: List = []
        for i in range(num_layers):
            layers.append(EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            ))
        self.layers = nn.ModuleList(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: torch.Tensor, full_image_embedding: torch.Tensor):
        # print(f"positional embedding shape: {self.pos_embedding.shape} ")
        x = x + self.pos_embedding
        # print(f"x shape after positional embedding is added: {x.shape}")
        for encoder_block in self.layers:
            x = encoder_block(self.dropout(x))
            # print(f"x shape encoder_block pass: {x.shape}")
            x = torch.cat((x, full_image_embedding), 1)
            # print(f"x shape concat with full embedding: {x.shape}")
        return self.ln(x)


class IEViT(ModifiedViT):
    """
    Implements IEViT as per https://www.sciencedirect.com/science/article/pii/S0169260722005223
    """

    def __init__(self,
                 full_embedding: FullImageEmbedder,
                 image_size: int,
                 patch_size: int,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 mlp_dim: int,
                 dropout: float = 0.0,
                 attention_dropout: float = 0.0,
                 num_classes: int = 6,
                 representation_size: Optional[int] = None,
                 norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            num_classes,
            representation_size,
            norm_layer)

        self.full_embedding = full_embedding
        self.encoder = IEncoder(
            self.seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
            attention_dropout,
            norm_layer
        )

    def forward(self, x: torch.Tensor):
        # Extract full image embedding
        full_image_emb = self.full_embedding(x)
        # print(f"Concurrent embedded x shape: {full_image_emb.shape}")
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # print(f"x shape after process_input: {x.shape}")
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # print(f"x shape after tokenization (passed to the encoder): {x.shape}")
        x = self.encoder(x, full_image_emb)

        # Classifier "token" as used by standard language architectures
        x = x[:, 0]

        x = self.heads(x)

        return x


def IEViT_tester():
    train_dataset = ImageDataset(
        Path(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data\X_train.npy"),
        Path(r"C:\Users\User\Desktop\University\Y2\Q3\Data Challenge 1\JBG040-Group13\dc1\data\Y_train.npy"))
    x = train_dataset[0:2][0]

    embedder = FullImageEmbedder(768)
    ie = IEViT(image_size=128, patch_size=16, num_layers=12, hidden_dim=768, mlp_dim=3072, num_heads=12,
               full_embedding=embedder,
               num_classes=6)
    print(ie(x).shape)
    # print(ie(x.view(-1, 1, 128, 128)).shape)

# IEViT_tester()
