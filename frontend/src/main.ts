import { createApp } from "vue"

import App from "./App.vue"
import { httpSearchService } from "./services/httpSearchService"
import { mockSearchService } from "./services/mockSearchService"
import { searchServiceKey } from "./services/searchService"
import "./styles/tokens.css"
import "./styles/global.css"

const searchService =
  import.meta.env.VITE_USE_MOCK_SEARCH === "true"
    ? mockSearchService
    : httpSearchService

createApp(App)
  .provide(searchServiceKey, searchService)
  .mount("#app")
