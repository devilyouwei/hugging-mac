import { describe, expect, it } from "vitest"
import { selectPalmRuntime } from "./config"
import type { ResourceStatus } from "@/vision/types"

const resource = (artifacts: ResourceStatus["artifacts"]): ResourceStatus => ({ model_id: "hand", revision: "1", variant: "float", default_runtime: null, variants: [], artifacts })
const artifact = (runtime: string, available: boolean): ResourceStatus["artifacts"][number] => ({ artifact_id: runtime, format: runtime, runtime, available, size_bytes: null })

describe("palm runtime selection", () => {
  it("prefers Core ML", () => expect(selectPalmRuntime(resource([artifact("onnx", true), artifact("coreml", true)]))).toBe("coreml"))
  it("falls back to ONNX", () => expect(selectPalmRuntime(resource([artifact("coreml", false), artifact("onnx", true)]))).toBe("onnx"))
  it("returns empty when no palm runtime is ready", () => expect(selectPalmRuntime(resource([]))).toBe(""))
})
