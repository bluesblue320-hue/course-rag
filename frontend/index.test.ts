import { readFileSync } from "node:fs"
import { resolve } from "node:path"

import { describe, expect, it } from "vitest"

describe("document language", () => {
  it("declares Simplified Chinese as the document language", () => {
    const indexHtml = readFileSync(resolve(process.cwd(), "index.html"), "utf8")

    expect(indexHtml).toMatch(/<html\s+lang="zh-CN">/)
  })
})
