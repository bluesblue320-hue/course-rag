import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import type { AskState } from "../types/ask"
import AskStatus from "./AskStatus.vue"

function errorState(code?: string, message = "安全的通用错误"): AskState {
  return {
    status: "error",
    question: "问题",
    message,
    code,
  }
}

describe("AskStatus", () => {
  it("announces loading politely", () => {
    const wrapper = mount(AskStatus, {
      props: { state: { status: "loading", question: "问题" } },
    })

    expect(wrapper.get('[role="status"]').text()).toContain(
      "正在检索课程资料并生成回答",
    )
    expect(wrapper.get('[role="status"]').attributes("aria-live")).toBe("polite")
  })

  it.each([
    [
      "LLM_NOT_CONFIGURED",
      "问答服务尚未完成配置，你仍可以切换到“语义检索”查看相关原文。",
    ],
    ["GENERATION_FAILED", "回答生成失败，请稍后重试。"],
    ["NETWORK_ERROR", "无法连接后端服务，请确认 FastAPI 已启动。"],
    [undefined, "安全的通用错误"],
  ])("shows the safe message for %s", (code, expected) => {
    const wrapper = mount(AskStatus, { props: { state: errorState(code) } })

    expect(wrapper.get('[role="alert"]').text()).toContain(expected)
  })

  it("emits retry from an accessible error alert", async () => {
    const wrapper = mount(AskStatus, {
      props: { state: errorState("GENERATION_FAILED") },
    })

    expect(wrapper.get('[role="alert"]').attributes("role")).toBe("alert")
    await wrapper.get("button").trigger("click")
    expect(wrapper.emitted("retry")).toHaveLength(1)
  })

  it("renders nothing while idle or successful", async () => {
    const wrapper = mount(AskStatus, {
      props: { state: { status: "idle" } },
    })
    expect(wrapper.find("div").exists()).toBe(false)
  })
})
