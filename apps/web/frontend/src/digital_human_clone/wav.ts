export function encodeWave(chunks: Float32Array[], sampleRate: number): Blob {
  const count = chunks.reduce((total, chunk) => total + chunk.length, 0)
  const buffer = new ArrayBuffer(44 + count * 2)
  const view = new DataView(buffer)
  write(view, 0, "RIFF")
  view.setUint32(4, 36 + count * 2, true)
  write(view, 8, "WAVE")
  write(view, 12, "fmt ")
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  write(view, 36, "data")
  view.setUint32(40, count * 2, true)
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

function write(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index))
  }
}
