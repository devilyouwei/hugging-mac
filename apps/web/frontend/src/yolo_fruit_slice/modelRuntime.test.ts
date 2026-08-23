import { describe, expect, it } from "vitest"

import type { ResourceStatus } from "@/vision/types"
import { availableGameRuntimes, firstAvailableModel } from "./modelRuntime"

function resource(available: Array<"coreml" | "pytorch-mps" | "onnx">): ResourceStatus {
  return {
    model_id: "ultralytics/yolov8-pose",
    revision: "test",
    variant: "n",
    default_runtime: "coreml",
    variants: [],
    artifacts: [
      { artifact_id: "source", format: "pytorch", runtime: "pytorch-mps", available: available.includes("pytorch-mps"), size_bytes: null },
      { artifact_id: "coreml", format: "coreml", runtime: "coreml", available: available.includes("coreml"), size_bytes: null },
      { artifact_id: "onnx", format: "onnx", runtime: "onnx", available: available.includes("onnx"), size_bytes: null },
    ],
  }
}

describe("availableGameRuntimes", () => {
  it("keeps the artifact order returned by the model manifest", () => {
    expect(availableGameRuntimes(resource(["pytorch-mps", "coreml"]))).toEqual(["pytorch-mps", "coreml"])
  })

  it("falls back to PyTorch MPS when Core ML is absent", () => {
    expect(availableGameRuntimes(resource(["pytorch-mps"]))).toEqual(["pytorch-mps"])
  })

  it("accepts ONNX as a loadable game runtime", () => {
    expect(availableGameRuntimes(resource(["onnx"]))).toEqual(["onnx"])
  })
})

describe("firstAvailableModel", () => {
  it("selects the first available variant and runtime combination", () => {
    const missing = resource([])
    const ready = { ...resource(["coreml"]), variant: "s" }
    expect(firstAvailableModel([missing, ready])).toEqual({ variant: "s", runtime: "coreml" })
  })

  it("returns null only when no supported runtime is available", () => {
    expect(firstAvailableModel([resource([])])).toBeNull()
  })
})
