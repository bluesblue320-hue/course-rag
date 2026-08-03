import { mount } from "@vue/test-utils"
import { describe, expect, it } from "vitest"

import type { AskResponse } from "../types/ask"
import AnswerCard from "./AnswerCard.vue"

const response: AskResponse = {
  question: "Service 层负责什么？",
  answer: "Service 层负责核心业务逻辑。[来源1]",
  retrieval_elapsed_ms: 12.4,
  generation_elapsed_ms: 680.7,
  total_elapsed_ms: 693.1,
  embedding_model: "embedding-model",
  llm_model: "llm-model",
  sources: [],
}

describe("AnswerCard", () => {
  it("shows the question and answer", () => {
    const wrapper = mount(AnswerCard, { props: { response } })

    expect(wrapper.text()).toContain("AI 回答")
    expect(wrapper.text()).toContain(response.question)
    expect(wrapper.get(".answer-text").text()).toBe(response.answer)
  })

  it("shows all timings with their original precision", () => {
    const wrapper = mount(AnswerCard, { props: { response } })

    expect(wrapper.text()).toContain("12.4 ms")
    expect(wrapper.text()).toContain("680.7 ms")
    expect(wrapper.text()).toContain("693.1 ms")
  })

  it("shows both model names", () => {
    const wrapper = mount(AnswerCard, { props: { response } })

    expect(wrapper.text()).toContain("embedding-model")
    expect(wrapper.text()).toContain("llm-model")
  })

  it("renders answers as escaped plain text instead of HTML", () => {
    const wrapper = mount(AnswerCard, {
      props: { response: { ...response, answer: "<strong>不要执行</strong>" } },
    })

    expect(wrapper.get(".answer-text").text()).toBe("<strong>不要执行</strong>")
    expect(wrapper.get(".answer-text").find("strong").exists()).toBe(false)
    expect(wrapper.get(".answer-text").html()).toContain("&lt;strong&gt;")
  })
})
