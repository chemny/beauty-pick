export function uploadPhoto<T>(form: FormData, token: string, progress: (percent: number) => void, processing: () => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest()
    request.open('POST', '/api/assets')
    request.setRequestHeader('x-portrait-token', token)
    request.timeout = 120_000
    request.upload.onprogress = event => {
      if (event.lengthComputable && event.total > 0) progress(Math.min(100, Math.round(event.loaded / event.total * 100)))
    }
    request.upload.onload = () => { progress(100); processing() }
    request.onload = () => {
      let body: { detail?: unknown }
      try { body = JSON.parse(request.responseText) } catch { reject(new Error('上传响应异常，请重试')); return }
      if (request.status < 200 || request.status >= 300) {
        reject(new Error(typeof body?.detail === 'string' ? body.detail : '上传失败，请重试'))
        return
      }
      resolve(body as T)
    }
    request.onerror = () => reject(new Error('网络连接中断，请重试'))
    request.ontimeout = () => reject(new Error('上传等待超时，请重试'))
    request.onabort = () => reject(new Error('上传已取消'))
    request.send(form)
  })
}
