"""BiSeNetV2 architecture for real-time semantic segmentation.

Implements the two-branch design from Yu et al., "BiSeNet V2: Bilateral
Network with Guided Aggregation for Real-Time Semantic Segmentation"
(IJCV 2021): a shallow, wide Detail Branch that preserves spatial precision
(edges, thin structures) at 1/8 resolution, and a deep, narrow Semantic
Branch that builds a large receptive field down to 1/32 resolution using
depthwise-separable "Gather-and-Expand" blocks (a MobileNetV2-style inverted
residual). A Bilateral Guided Aggregation (BGA) layer fuses the two before
a lightweight segmentation head produces the final per-pixel logits.

This was chosen over ENet (faster but noticeably lower accuracy — it
sacrifices representational capacity to hit extreme speed) and a small
U-Net (no built-in mechanism to stay fast at inference; shrinking it enough
to hit a real-time target usually costs about as much accuracy as ENet's
ceiling anyway) because it is the only one of the three specifically
designed to deliver strong accuracy *while remaining real-time* rather than
trading one for the other.

Written independently from the paper's description, not copied from any
specific open-source implementation. Channel widths are simplified from the
paper's exact configuration for a smaller, faster-to-train model appropriate
to this assignment's timeline and dataset size — the two-branch design and
BGA fusion (the architecture's actual defining contribution) are preserved.
Auxiliary "booster" training heads (see BiSeNetV2.forward_with_aux) are
included for deeper supervision during training and add zero inference-time
cost, since they're simply not computed on the plain inference path.
"""

import torch
from torch import nn
from torch.nn import functional as F


def _conv_bn_act(
    in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, groups: int = 1
) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU, the basic building block used throughout."""
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DetailBranch(nn.Module):
    """Shallow, wide branch preserving spatial detail at 1/8 resolution.

    Three stages, each halving resolution once: input -> 1/2 -> 1/4 -> 1/8.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            _conv_bn_act(3, 64, stride=2),
            _conv_bn_act(64, 64, stride=1),
        )
        self.stage2 = nn.Sequential(
            _conv_bn_act(64, 64, stride=2),
            _conv_bn_act(64, 64, stride=1),
            _conv_bn_act(64, 64, stride=1),
        )
        self.stage3 = nn.Sequential(
            _conv_bn_act(64, 128, stride=2),
            _conv_bn_act(128, 128, stride=1),
            _conv_bn_act(128, 128, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x  # (B, 128, H/8, W/8)


class StemBlock(nn.Module):
    """First downsampling block of the Semantic Branch: input -> 1/4 resolution."""

    def __init__(self, out_ch: int = 16) -> None:
        super().__init__()
        self.conv = _conv_bn_act(3, out_ch, stride=2)
        self.branch_conv = nn.Sequential(
            _conv_bn_act(out_ch, out_ch // 2, kernel_size=1, stride=1),
            _conv_bn_act(out_ch // 2, out_ch, stride=2),
        )
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fuse = _conv_bn_act(out_ch * 2, out_ch, kernel_size=3, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        branch_a = self.branch_conv(x)
        branch_b = self.branch_pool(x)
        return self.fuse(torch.cat([branch_a, branch_b], dim=1))


class GELayer(nn.Module):
    """Gather-and-Expand layer: a depthwise-separable inverted residual block.

    Simplified from the paper's version (single depthwise stage rather than
    two) but keeps the core idea: cheaply expand channels with a depthwise
    conv for a wider receptive field, then project back down, with a
    residual connection. Set ``stride=2`` to also halve resolution, in which
    case the shortcut path is a depthwise+pointwise conv rather than an
    identity (since the main path's spatial size no longer matches the
    input's).
    """

    def __init__(self, in_ch: int, out_ch: int, expansion: int = 6, stride: int = 1) -> None:
        super().__init__()
        self.stride = stride
        mid_ch = in_ch * expansion

        self.conv1 = _conv_bn_act(in_ch, in_ch, kernel_size=3, stride=1)
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        if stride == 2 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1,
                          groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.depthwise(out)
        out = self.project(out)
        return self.act(out + residual)


class CEBlock(nn.Module):
    """Context Embedding block: injects global image-level context via
    global average pooling, added back into the local feature map.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(channels)
        self.conv_gap = _conv_bn_act(channels, channels, kernel_size=1, stride=1)
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gap = x.mean(dim=(2, 3), keepdim=True)  # global average pool to 1x1
        gap = self.bn(gap)
        gap = self.conv_gap(gap)
        fused = x + gap  # broadcast global context over every spatial position
        return self.conv_out(fused)


class SemanticBranch(nn.Module):
    """Deep, narrow branch building semantic context down to 1/32 resolution.

    Stages: Stem (1/4) -> S3 (1/8) -> S4 (1/16) -> S5 + context embedding (1/32).
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = StemBlock(out_ch=16)
        self.stage3 = nn.Sequential(
            GELayer(16, 32, stride=2),
            GELayer(32, 32, stride=1),
        )
        self.stage4 = nn.Sequential(
            GELayer(32, 64, stride=2),
            GELayer(64, 64, stride=1),
        )
        self.stage5 = nn.Sequential(
            GELayer(64, 128, stride=2),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
            GELayer(128, 128, stride=1),
        )
        self.context = CEBlock(128)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stage3_out = self.stem(x)
        stage3_out = self.stage3(stage3_out)
        stage4_out = self.stage4(stage3_out)
        stage5_out = self.stage5(stage4_out)
        context_out = self.context(stage5_out)
        # Intermediate stage outputs are returned alongside the final one so
        # BiSeNetV2 can optionally attach auxiliary training heads to them
        # (deeper supervision — see BiSeNetV2.forward_with_aux). Only
        # context_out is used for the main prediction path.
        return context_out, stage3_out, stage4_out, stage5_out


class BGALayer(nn.Module):
    """Bilateral Guided Aggregation: fuses the Detail and Semantic branches.

    Each branch is used to gate (modulate) the other before combining, so
    the fine-grained detail branch and the context-heavy semantic branch
    reinforce rather than simply average out each other's information.
    """

    def __init__(self, channels: int = 128) -> None:
        super().__init__()
        self.detail_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
        )
        self.semantic_gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.fuse = _conv_bn_act(channels, channels, kernel_size=3, stride=1)

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> torch.Tensor:
        semantic_up = F.interpolate(
            semantic, size=detail.shape[2:], mode="bilinear", align_corners=False
        )
        detail_gated = detail * torch.sigmoid(self.semantic_gate(semantic_up))
        semantic_gated = semantic_up * torch.sigmoid(self.detail_gate(detail))
        return self.fuse(detail_gated + semantic_gated)


class SegmentHead(nn.Module):
    """Lightweight head: refine the fused feature map, then upsample to
    the original input resolution.
    """

    def __init__(self, in_ch: int, num_classes: int, upsample_factor: int = 8) -> None:
        super().__init__()
        self.refine = _conv_bn_act(in_ch, in_ch, kernel_size=3, stride=1)
        self.classify = nn.Conv2d(in_ch, num_classes, kernel_size=1)
        self.upsample_factor = upsample_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.refine(x)
        x = self.classify(x)
        return F.interpolate(
            x, scale_factor=self.upsample_factor, mode="bilinear", align_corners=False
        )


class BiSeNetV2(nn.Module):
    """Full BiSeNetV2: Detail Branch + Semantic Branch -> BGA -> SegmentHead.

    Input and output spatial size match exactly (the head upsamples back up
    to the Detail Branch's resolution's original scale), matching the
    contract SemanticSegmenter.predict() expects — no separate resize step
    needed by the caller between model output and the input resolution.

    Also exposes three auxiliary segmentation heads attached to the
    Semantic Branch's intermediate stages (the paper's "booster" heads),
    used only via :meth:`forward_with_aux` during training for deeper
    supervision — they add extra loss terms that help gradient flow into
    the semantic branch's earlier layers, at zero inference-time cost since
    they're simply not computed when the plain :meth:`forward` path is
    used (as SemanticSegmenter.predict() always does).
    """

    def __init__(self, num_classes: int = 19) -> None:
        super().__init__()
        self.detail_branch = DetailBranch()
        self.semantic_branch = SemanticBranch()
        self.bga = BGALayer(channels=128)
        self.head = SegmentHead(in_ch=128, num_classes=num_classes, upsample_factor=8)

        # Auxiliary heads on the semantic branch's stage3/stage4/stage5
        # outputs (channels 32/64/128, at 1/8, 1/16, 1/32 resolution
        # respectively). Each is a lightweight 1x1 classifier followed by
        # an upsample back to full input resolution, matching the paper's
        # "Seg Head" design used for its booster heads.
        self.aux_head3 = SegmentHead(in_ch=32, num_classes=num_classes, upsample_factor=8)
        self.aux_head4 = SegmentHead(in_ch=64, num_classes=num_classes, upsample_factor=16)
        self.aux_head5 = SegmentHead(in_ch=128, num_classes=num_classes, upsample_factor=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        detail = self.detail_branch(x)                     # (B, 128, H/8, W/8)
        semantic, _, _, _ = self.semantic_branch(x)         # (B, 128, H/32, W/32)
        fused = self.bga(detail, semantic)                  # (B, 128, H/8, W/8)
        return self.head(fused)                             # (B, num_classes, H, W)

    def forward_with_aux(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training-only forward pass: main logits plus three auxiliary outputs.

        Use this in the training loop instead of the plain ``forward`` /
        ``model(x)`` call to get deeper supervision; sum the main loss with
        (typically downweighted) losses on each auxiliary output. Not used
        by SemanticSegmenter — inference always uses the cheaper plain
        ``forward`` path, since the aux heads exist purely to improve what
        gradients reach the semantic branch during training.
        """
        detail = self.detail_branch(x)
        semantic, stage3_out, stage4_out, stage5_out = self.semantic_branch(x)
        fused = self.bga(detail, semantic)
        main_out = self.head(fused)
        aux3 = self.aux_head3(stage3_out)
        aux4 = self.aux_head4(stage4_out)
        aux5 = self.aux_head5(stage5_out)
        return main_out, aux3, aux4, aux5
