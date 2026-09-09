import { describe, expect, it } from "vitest"

import { preferredParserModelId } from "./modelSelection"
import type { DocumentRecord, ParserModel } from "./types"

const models: ParserModel[] = [
  { model_id: "first", name: "First", description: "", variant: "4bit", runtime: "mlx", ready: false, source_url: null },
  { model_id: "ready", name: "Ready", description: "", variant: "8bit", runtime: "mlx", ready: true, source_url: null },
]

describe("preferredParserModelId", () => {
  it("restores the model used by the latest Markdown job", () => {
    const document = { jobs: { markdown: { model_id: "first" } } } as unknown as DocumentRecord
    expect(preferredParserModelId(models, document, "ready")).toBe("first")
  })

  it("uses the remembered choice and otherwise the first ready model", () => {
    expect(preferredParserModelId(models, null, "first")).toBe("first")
    expect(preferredParserModelId(models)).toBe("ready")
  })
})
