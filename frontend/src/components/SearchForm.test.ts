import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import SearchForm from "./SearchForm.vue"

describe("SearchForm", () => {
  it("shows a local validation error for a blank query", async () => {
    const wrapper = mount(SearchForm)

    await wrapper.get("form").trigger("submit")

    expect(wrapper.get('[role="alert"]').text()).toBe("请输入问题")
    expect(wrapper.emitted("submit")).toBeUndefined()
  })

  it("emits a trimmed query from the submit button", async () => {
    const wrapper = mount(SearchForm)
    await wrapper.get("textarea").setValue(" 业务逻辑应该写在哪一层？ ")

    await wrapper.get("form").trigger("submit")

    expect(wrapper.emitted("submit")).toEqual([["业务逻辑应该写在哪一层？"]])
  })

  it("submits with Enter", async () => {
    const wrapper = mount(SearchForm)
    await wrapper.get("textarea").setValue("业务逻辑")

    await wrapper.get("textarea").trigger("keydown", {
      key: "Enter",
      shiftKey: false,
    })

    expect(wrapper.emitted("submit")).toEqual([["业务逻辑"]])
  })

  it("does not submit with Shift+Enter", async () => {
    const wrapper = mount(SearchForm)
    await wrapper.get("textarea").setValue("第一行")

    await wrapper.get("textarea").trigger("keydown", {
      key: "Enter",
      shiftKey: true,
    })

    expect(wrapper.emitted("submit")).toBeUndefined()
  })

  it("disables both controls while loading", () => {
    const wrapper = mount(SearchForm, {
      props: { loading: true },
    })

    expect(wrapper.get("textarea").attributes("disabled")).toBeDefined()
    expect(wrapper.get("button").attributes("disabled")).toBeDefined()
    expect(wrapper.get("button").text()).toBe("检索中…")
  })
})
