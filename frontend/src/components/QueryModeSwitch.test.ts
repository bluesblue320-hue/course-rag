import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import QueryModeSwitch from "./QueryModeSwitch.vue"

describe("QueryModeSwitch", () => {
  it("renders all three modes with ask selected", () => {
    const wrapper = mount(QueryModeSwitch, { props: { modelValue: "ask" } })
    const buttons = wrapper.findAll("button")

    expect(buttons).toHaveLength(3)
    expect(wrapper.text()).toContain("智能问答")
    expect(wrapper.text()).toContain("语义检索")
    expect(wrapper.text()).toContain("知识库管理")
    expect(buttons[0].attributes("aria-pressed")).toBe("true")
    expect(buttons[1].attributes("aria-pressed")).toBe("false")
    expect(buttons[2].attributes("aria-pressed")).toBe("false")
  })

  it("emits search when the retrieval button is clicked", async () => {
    const wrapper = mount(QueryModeSwitch, { props: { modelValue: "ask" } })

    await wrapper.findAll("button")[1].trigger("click")

    expect(wrapper.emitted("update:modelValue")).toEqual([["search"]])
  })

  it("emits documents when the knowledge base button is clicked", async () => {
    const wrapper = mount(QueryModeSwitch, { props: { modelValue: "ask" } })

    await wrapper.findAll("button")[2].trigger("click")

    expect(wrapper.emitted("update:modelValue")).toEqual([["documents"]])
  })

  it("does not switch while disabled", async () => {
    const wrapper = mount(QueryModeSwitch, {
      props: { modelValue: "ask", disabled: true },
    })

    await wrapper.findAll("button")[1].trigger("click")

    expect(wrapper.emitted("update:modelValue")).toBeUndefined()
    expect(wrapper.findAll("button").every((button) => "disabled" in button.attributes())).toBe(
      true,
    )
  })

  it("uses native keyboard-operable button semantics", () => {
    const wrapper = mount(QueryModeSwitch, { props: { modelValue: "search" } })

    for (const button of wrapper.findAll("button")) {
      expect(button.element.tagName).toBe("BUTTON")
      expect(button.attributes("type")).toBe("button")
    }
    expect(wrapper.findAll("button")[1].attributes("aria-pressed")).toBe("true")
  })
})
