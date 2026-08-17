import { describe, expect, it } from "vitest"

import type { ResourceStatus } from "@/vision/types"
import { availableGameRuntimes } from "./modelRuntime"

function resource(available: Array<"coreml" | "pytorch-mps">): ResourceStatus {
  return {
    model_id: "ultralytics/yolov8-pose",
    revision: "test",
    variant: "n",
    default_runtime: "coreml",
    variants: [],
    artifacts: [
      { artifact_id: "source", format: "pytorch", runtime: "pytorch-mps", available: available.includes("pytorch-mps"), size_bytes: null },
      { artifact_id: "coreml", format: "coreml", runtime: "coreml", available: available.includes("coreml"), size_bytes: null },
    ],
  }
}

describe("availableGameRuntimes", () => {
  it("prefers Core ML before PyTorch MPS", () => {
    expect(availableGameRuntimes(resource(["pytorch-mps", "coreml"]))).toEqual(["coreml", "pytorch-mps"])
  })

  it("falls back to PyTorch MPS when Core ML is absent", () => {
    expect(availableGameRuntimes(resource(["pytorch-mps"]))).toEqual(["pytorch-mps"])
  })
})
