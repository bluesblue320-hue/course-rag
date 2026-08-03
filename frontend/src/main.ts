import { createApp } from "vue"

import App from "./App.vue"
import { askServiceKey } from "./services/askService"
import { httpAskService } from "./services/httpAskService"
import { httpSearchService } from "./services/httpSearchService"
import { mockAskService } from "./services/mockAskService"
import { mockSearchService } from "./services/mockSearchService"
import { searchServiceKey } from "./services/searchService"
import "./styles/tokens.css"
import "./styles/global.css"

const useMockSearch = import.meta.env.VITE_USE_MOCK_SEARCH === "true"
const useMockAsk = import.meta.env.VITE_USE_MOCK_ASK === "true"

const searchService = useMockSearch ? mockSearchService : httpSearchService
const askService = useMockAsk ? mockAskService : httpAskService

createApp(App)
  .provide(searchServiceKey, searchService)
  .provide(askServiceKey, askService)
  .mount("#app")
