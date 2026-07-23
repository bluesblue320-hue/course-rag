import { createApp } from "vue"

import App from "./App.vue"
import { mockSearchService } from "./services/mockSearchService"
import { searchServiceKey } from "./services/searchService"
import "./styles/tokens.css"
import "./styles/global.css"

createApp(App)
  .provide(searchServiceKey, mockSearchService)
  .mount("#app")
