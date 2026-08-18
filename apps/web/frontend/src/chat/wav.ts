export function encodePcm16(chunks: Float32Array[]): ArrayBuffer {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0)
  const output = new ArrayBuffer(sampleCount * 2)
  const view = new DataView(output)
  let offset = 0
  for (const chunk of chunks) {
    for (const sample of chunk) {
      const clamped = Math.max(-1, Math.min(1, sample))
      view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true)
      offset += 2
    }
  }
  return output
}

interface WavParts {
  format: Uint8Array
  data: Uint8Array
}

export async function mergeWavBlobs(blobs: Blob[]): Promise<Blob> {
  if (!blobs.length) throw new Error("No WAV audio to merge")
  if (blobs.length === 1) return blobs[0]
  const parts = await Promise.all(blobs.map(async (blob) => parseWav(await blob.arrayBuffer())))
  const format = parts[0]!.format
  if (parts.some((part) => !equalBytes(part.format, format))) {
    throw new Error("TTS segments use incompatible WAV formats")
  }
  const dataLength = parts.reduce((total, part) => total + part.data.byteLength, 0)
  const formatPadding = format.byteLength % 2
  const output = new ArrayBuffer(12 + 8 + format.byteLength + formatPadding + 8 + dataLength)
  const view = new DataView(output)
  const bytes = new Uint8Array(output)
  writeAscii(bytes, 0, "RIFF")
  view.setUint32(4, output.byteLength - 8, true)
  writeAscii(bytes, 8, "WAVE")
  writeAscii(bytes, 12, "fmt ")
  view.setUint32(16, format.byteLength, true)
  bytes.set(format, 20)
  const dataHeader = 20 + format.byteLength + formatPadding
  writeAscii(bytes, dataHeader, "data")
  view.setUint32(dataHeader + 4, dataLength, true)
  let offset = dataHeader + 8
  for (const part of parts) {
    bytes.set(part.data, offset)
    offset += part.data.byteLength
  }
  return new Blob([output], { type: "audio/wav" })
}

function parseWav(buffer: ArrayBuffer): WavParts {
  const bytes = new Uint8Array(buffer)
  const view = new DataView(buffer)
  if (readAscii(bytes, 0, 4) !== "RIFF" || readAscii(bytes, 8, 4) !== "WAVE") {
    throw new Error("TTS response is not a WAV file")
  }
  let format: Uint8Array | null = null
  let data: Uint8Array | null = null
  for (let offset = 12; offset + 8 <= buffer.byteLength;) {
    const id = readAscii(bytes, offset, 4)
    const size = view.getUint32(offset + 4, true)
    const start = offset + 8
    const end = Math.min(start + size, buffer.byteLength)
    if (id === "fmt ") format = bytes.slice(start, end)
    if (id === "data") data = bytes.slice(start, end)
    offset = start + size + (size % 2)
  }
  if (!format || !data) throw new Error("TTS WAV is missing format or audio data")
  return { format, data }
}

function readAscii(bytes: Uint8Array, offset: number, length: number): string {
  return String.fromCharCode(...bytes.subarray(offset, offset + length))
}

function writeAscii(bytes: Uint8Array, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) bytes[offset + index] = value.charCodeAt(index)
}

function equalBytes(left: Uint8Array, right: Uint8Array): boolean {
  return left.byteLength === right.byteLength && left.every((value, index) => value === right[index])
}
