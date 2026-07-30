export function encodeWave(
  chunks: Float32Array[],
  sampleRate: number,
): Blob {
  const sampleCount = chunks.reduce((total, chunk) => total + chunk.length, 0)
  const buffer = new ArrayBuffer(44 + sampleCount * 2)
  const view = new DataView(buffer)

  writeText(view, 0, "RIFF")
  view.setUint32(4, 36 + sampleCount * 2, true)
  writeText(view, 8, "WAVE")
  writeText(view, 12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeText(view, 36, "data")
  view.setUint32(40, sampleCount * 2, true)

  let offset = 44
  for (const chunk of chunks) {
    for (const sample of chunk) {
      const clipped = Math.max(-1, Math.min(1, sample))
      view.setInt16(offset, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true)
      offset += 2
    }
  }
  return new Blob([buffer], { type: "audio/wav" })
}

function writeText(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}
