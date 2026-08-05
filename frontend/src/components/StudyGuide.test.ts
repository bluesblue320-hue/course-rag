import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import StudyGuide from "./StudyGuide.vue"

describe("StudyGuide", () => {
  it("explains the learning workflow, reliability checks, and material limits", () => {
    const wrapper = mount(StudyGuide)

    expect(wrapper.get("#guide-lead-title").text()).toContain("先准备资料")
    expect(wrapper.findAll(".workflow-list li")).toHaveLength(3)
    expect(wrapper.text()).toContain("0.35")
    expect(wrapper.text()).toContain("文本型 PDF")
    expect(wrapper.text()).toContain("扫描 PDF 的 OCR")
  })
})
