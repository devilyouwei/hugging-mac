export async function captureVideoFrame(
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement,
  filename: string,
  {
    maxDimension = 640,
    quality = 0.82,
  }: {
    maxDimension?: number
    quality?: number
  } = {},
): Promise<File> {
  if (!video.videoWidth || !video.videoHeight) {
    throw new Error("视频画面尚未准备好")
  }

  const scale = Math.min(1, maxDimension / Math.max(video.videoWidth, video.videoHeight))
  const width = Math.max(1, Math.round(video.videoWidth * scale))
  const height = Math.max(1, Math.round(video.videoHeight * scale))
  if (canvas.width !== width) canvas.width = width
  if (canvas.height !== height) canvas.height = height
  const context = canvas.getContext("2d", {
    alpha: false,
    desynchronized: true,
  })
  if (!context) throw new Error("浏览器无法创建视频帧画布")
  context.drawImage(video, 0, 0, width, height)

  const blob = await new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (value) => (value ? resolve(value) : reject(new Error("视频帧编码失败"))),
      "image/jpeg",
      quality,
    )
  })
  return new File([blob], filename, { type: blob.type })
}

export function wait(milliseconds: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, milliseconds))
}

export async function waitForNextVideoFrame(
  video: HTMLVideoElement,
): Promise<boolean> {
  if (video.ended) return false
  if (typeof video.requestVideoFrameCallback !== "function") {
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
    return !video.ended
  }

  return new Promise<boolean>((resolve) => {
    let settled = false
    const finish = (hasFrame: boolean) => {
      if (settled) return
      settled = true
      video.removeEventListener("ended", handleEnded)
      resolve(hasFrame)
    }
    const handleEnded = () => finish(false)
    video.addEventListener("ended", handleEnded, { once: true })
    video.requestVideoFrameCallback(() => finish(true))
  })
}

export async function seekVideo(
  video: HTMLVideoElement,
  time: number,
): Promise<void> {
  const target = Math.min(Math.max(time, 0), video.duration)
  if (Math.abs(video.currentTime - target) < 0.001) {
    await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()))
    return
  }

  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      video.removeEventListener("seeked", handleSeeked)
      video.removeEventListener("error", handleError)
    }
    const handleSeeked = () => {
      cleanup()
      resolve()
    }
    const handleError = () => {
      cleanup()
      reject(new Error("浏览器无法定位到指定视频帧"))
    }
    video.addEventListener("seeked", handleSeeked, { once: true })
    video.addEventListener("error", handleError, { once: true })
    video.currentTime = target
  })
}
