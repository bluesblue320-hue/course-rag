import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import DocumentUpload from "./DocumentUpload.vue"

async function selectFile(
  wrapper: ReturnType<typeof mount>,
  filename: string,
  content: string,
): Promise<void> {
  const file = new File([content], filename, { type: "text/plain" })
  const input = wrapper.get('input[type="file"]').element as HTMLInputElement
  Object.defineProperty(input, "files", { value: [file] })
  await wrapper.get('input[type="file"]').trigger("change")
}

describe("DocumentUpload", () => {
  it("shows supported formats and the size limit", () => {
    const wrapper = mount(DocumentUpload, {
      props: { status: "idle", maxUploadBytes: 10 * 1024 * 1024 },
    })

    expect(wrapper.text()).toContain(".txt、.md 和文本型 .pdf")
    expect(wrapper.text()).toContain("最大 10 MB")
    expect(wrapper.get('input[type="file"]').attributes("accept")).toBe(
      ".txt,.md,.pdf",
    )
  })

  it("selects a file and enables the upload button", async () => {
    const wrapper = mount(DocumentUpload, { props: { status: "idle" } })

    expect(
      (wrapper.get("button[type='submit']").attributes("disabled") !== undefined),
    ).toBe(true)

    await selectFile(wrapper, "notes.txt", "alpha content")

    expect(wrapper.text()).toContain("notes.txt")
    expect(
      (wrapper.get("button[type='submit']").attributes("disabled") !== undefined),
    ).toBe(false)
  })

  it("emits the selected file and disables the button while uploading", async () => {
    const wrapper = mount(DocumentUpload, {
      props: { status: "idle" },
    })
    await selectFile(wrapper, "notes.txt", "alpha content")

    await wrapper.get("form").trigger("submit")
    expect(wrapper.emitted("upload")).toHaveLength(1)

    await wrapper.setProps({ status: "uploading", uploading: true })
    expect(
      (wrapper.get("button[type='submit']").attributes("disabled") !== undefined),
    ).toBe(true)
    expect(wrapper.text()).toContain("上传中…")
  })

  it("clears the selection after a successful upload", async () => {
    const wrapper = mount(DocumentUpload, {
      props: { status: "uploading", uploading: true },
    })
    await selectFile(wrapper, "notes.txt", "alpha content")

    await wrapper.setProps({ status: "success", uploading: false })

    expect(wrapper.text()).toContain("上传成功")
    expect(wrapper.text()).not.toContain("notes.txt")
  })

  it("shows a stable error message and keeps the file for retry", async () => {
    const wrapper = mount(DocumentUpload, {
      props: { status: "idle" },
    })
    await selectFile(wrapper, "notes.txt", "alpha content")

    await wrapper.setProps({
      status: "error",
      message: "文档解析失败，请确认文件内容有效",
    })

    expect(wrapper.get('[role="alert"]').text()).toContain("文档解析失败")
    expect(wrapper.text()).toContain("notes.txt")
  })

  it("resets the status when a new file is selected after an error", async () => {
    const wrapper = mount(DocumentUpload, {
      props: { status: "error", message: "文档解析失败" },
    })

    await selectFile(wrapper, "other.txt", "content")

    expect(wrapper.emitted("resetStatus")).toHaveLength(1)
  })
})
