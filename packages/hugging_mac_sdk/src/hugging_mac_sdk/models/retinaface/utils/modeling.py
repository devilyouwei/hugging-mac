"""MobileNet0.25 RetinaFace architecture adapted from py-feat (MIT)."""

# mypy: disable-error-code="name-defined,misc"

from __future__ import annotations

from collections import OrderedDict
from typing import Any


def build_retinaface(torch: Any) -> Any:
    nn, functional = torch.nn, torch.nn.functional

    def conv_bn(inp: int, out: int, stride: int = 1, leaky: float = 0.0) -> Any:
        return nn.Sequential(
            nn.Conv2d(inp, out, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out),
            nn.LeakyReLU(negative_slope=leaky, inplace=True),
        )

    def conv_dw(inp: int, out: int, stride: int, leaky: float = 0.1) -> Any:
        return nn.Sequential(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.LeakyReLU(negative_slope=leaky, inplace=True),
            nn.Conv2d(inp, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out),
            nn.LeakyReLU(negative_slope=leaky, inplace=True),
        )

    class Body(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stage1 = nn.Sequential(
                conv_bn(3, 8, 2, 0.1),
                conv_dw(8, 16, 1),
                conv_dw(16, 32, 2),
                conv_dw(32, 32, 1),
                conv_dw(32, 64, 2),
                conv_dw(64, 64, 1),
            )
            self.stage2 = nn.Sequential(
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 128, 1),
                conv_dw(128, 128, 1),
                conv_dw(128, 128, 1),
                conv_dw(128, 128, 1),
            )
            self.stage3 = nn.Sequential(conv_dw(128, 256, 2), conv_dw(256, 256, 1))

        def forward(self, value: Any) -> OrderedDict[str, Any]:
            first = self.stage1(value)
            second = self.stage2(first)
            third = self.stage3(second)
            return OrderedDict((("1", first), ("2", second), ("3", third)))

    class FPN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output1 = conv_bn1x1(64, 64)
            self.output2 = conv_bn1x1(128, 64)
            self.output3 = conv_bn1x1(256, 64)
            self.merge1 = conv_bn(64, 64, leaky=0.1)
            self.merge2 = conv_bn(64, 64, leaky=0.1)

        def forward(self, values: Any) -> list[Any]:
            first, second, third = list(values.values())
            output1, output2, output3 = (
                self.output1(first),
                self.output2(second),
                self.output3(third),
            )
            output2 = self.merge2(
                output2 + functional.interpolate(output3, size=output2.shape[2:], mode="nearest")
            )
            output1 = self.merge1(
                output1 + functional.interpolate(output2, size=output1.shape[2:], mode="nearest")
            )
            return [output1, output2, output3]

    def conv_bn1x1(inp: int, out: int) -> Any:
        return nn.Sequential(
            nn.Conv2d(inp, out, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    class SSH(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv3X3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32))
            self.conv5X5_1 = conv_bn(64, 16, leaky=0.1)
            self.conv5X5_2 = nn.Sequential(
                nn.Conv2d(16, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)
            )
            self.conv7X7_2 = conv_bn(16, 16, leaky=0.1)
            self.conv7x7_3 = nn.Sequential(
                nn.Conv2d(16, 16, 3, 1, 1, bias=False), nn.BatchNorm2d(16)
            )

        def forward(self, value: Any) -> Any:
            three = self.conv3X3(value)
            five1 = self.conv5X5_1(value)
            five = self.conv5X5_2(five1)
            seven = self.conv7x7_3(self.conv7X7_2(five1))
            return functional.relu(torch.cat((three, five, seven), dim=1))

    class Head(nn.Module):
        def __init__(self, outputs: int) -> None:
            super().__init__()
            self.conv1x1 = nn.Conv2d(64, outputs * 2, 1)
            self.outputs = outputs

        def forward(self, value: Any) -> Any:
            result = self.conv1x1(value).permute(0, 2, 3, 1).contiguous()
            return result.view(result.shape[0], -1, self.outputs)

    class RetinaFace(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = Body()
            self.fpn = FPN()
            self.ssh1, self.ssh2, self.ssh3 = SSH(), SSH(), SSH()
            self.ClassHead = nn.ModuleList((Head(2), Head(2), Head(2)))
            self.BboxHead = nn.ModuleList((Head(4), Head(4), Head(4)))
            self.LandmarkHead = nn.ModuleList((Head(10), Head(10), Head(10)))

        def forward(self, value: Any) -> tuple[Any, Any, Any]:
            pyramid = self.fpn(self.body(value))
            features = (self.ssh1(pyramid[0]), self.ssh2(pyramid[1]), self.ssh3(pyramid[2]))
            boxes = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], 1)
            classes = torch.cat(
                [self.ClassHead[i](feature) for i, feature in enumerate(features)], 1
            )
            landmarks = torch.cat(
                [self.LandmarkHead[i](feature) for i, feature in enumerate(features)], 1
            )
            return boxes, functional.softmax(classes, dim=-1), landmarks

    return RetinaFace()
