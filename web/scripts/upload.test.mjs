import { test } from 'node:test'
import assert from 'node:assert/strict'
import { uploadPhoto } from '../src/upload.ts'

class Request {
  static last
  upload = {}
  headers = {}
  constructor() { Request.last = this }
  open(method, url) { this.method = method; this.url = url }
  setRequestHeader(key, value) { this.headers[key] = value }
  send(form) { this.form = form }
}
globalThis.XMLHttpRequest = Request

test('transfer progress stays pending while the server processes the uploaded photo', async () => {
  const progress = []; let processing = 0; let done = false
  const form = new FormData()
  const result = uploadPhoto(form, 'test-token', p => progress.push(p), () => processing++)
  result.then(() => { done = true })
  const request = Request.last
  assert.equal(request.url, '/api/assets')
  assert.equal(request.headers['x-portrait-token'], 'test-token')
  assert.equal(request.form, form)
  request.upload.onprogress({ lengthComputable: true, loaded: 25, total: 100 })
  assert.deepEqual(progress, [25])
  request.upload.onload()
  await Promise.resolve()
  assert.equal(processing, 1)
  assert.equal(done, false)
  assert.deepEqual(progress, [25, 100])
  request.status = 200; request.responseText = '{"id":"photo"}'; request.onload()
  assert.deepEqual(await result, { id: 'photo' })
})

test('unknown lengths do not invent a percentage', async () => {
  const progress = []
  const result = uploadPhoto(new FormData(), '', p => progress.push(p), () => {})
  Request.last.upload.onprogress({ lengthComputable: false, loaded: 1, total: 0 })
  assert.deepEqual(progress, [])
  Request.last.status = 200; Request.last.responseText = '{}'; Request.last.onload()
  await result
})

for (const event of ['onerror', 'ontimeout', 'onabort']) {
  test(event + ' rejects rather than leaving the upload waiting', async () => {
    const result = uploadPhoto(new FormData(), '', () => {}, () => {})
    Request.last[event]()
    await assert.rejects(result)
  })
}

test('server rejection and malformed responses are surfaced', async () => {
  let result = uploadPhoto(new FormData(), '', () => {}, () => {})
  Request.last.status = 413; Request.last.responseText = '{"detail":"照片过大"}'; Request.last.onload()
  await assert.rejects(result, /照片过大/)
  result = uploadPhoto(new FormData(), '', () => {}, () => {})
  Request.last.status = 200; Request.last.responseText = '<html>error</html>'; Request.last.onload()
  await assert.rejects(result, /响应异常/)
})
