import type { DocumentRecord, ParserModel } from "./types"

export function preferredParserModelId(
  models: ParserModel[],
  document?: DocumentRecord | null,
  rememberedId?: string,
): string {
  const candidates = [document?.jobs?.markdown?.model_id, rememberedId]
  for (const candidate of candidates) {
    if (candidate && models.some((model) => model.model_id === candidate)) return candidate
  }
  return models.find((model) => model.ready)?.model_id ?? models[0]?.model_id ?? ""
}
