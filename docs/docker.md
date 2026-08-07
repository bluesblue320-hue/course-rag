# Docker Compose 使用指南

本方案用于本地完整演示，并为后续单机云部署保留清晰的容器边界。它只封装现有 FastAPI、Vue 和文件系统持久化行为，不修改 API 路径、检索算法、问答逻辑或前端产品功能。

## 架构

```text
浏览器
  │ http://localhost:8080
  ▼
frontend 容器（Nginx + Vue 静态文件）
  ├── /                    → Vue SPA
  ├── /nginx-health        → Nginx 健康检查
  └── /api/*               → backend:8000/*（移除 /api 前缀）
                                  │
                                  ├── /health
                                  ├── /search
                                  ├── /ask
                                  └── /documents

db 容器（pgvector/pgvector:pg17）
  └── 只在 Compose 网络内暴露 5432；默认不映射宿主机端口
```

Compose 启动三个服务：

| 服务 | 内容 | 对外端口 | 健康检查 |
| --- | --- | --- | --- |
| `backend` | FastAPI、Embedding 模型和现有内存索引 | 不直接发布 | 容器内请求 `/health` |
| `frontend` | 多阶段构建的 Vue 静态文件和 Nginx 反向代理 | 默认 `8080` | 请求 `/nginx-health` |
| `db` | PostgreSQL 17 + pgvector 扩展（基础设施） | 不发布 | `pg_isready` |

后端只在 Compose 内部网络暴露 `8000`，浏览器继续使用现有 `/api/*` 路径。Nginx 转发时移除 `/api` 前缀，因此 FastAPI 的 `/health`、`/search`、`/ask` 和 `/documents` 路径均保持不变。`db` 服务默认是**准备好的基础设施**：基础 `compose.yaml` 保持 `VECTOR_STORE_BACKEND=memory`，后端不连接数据库，`db` 未启动也不影响后端运行。需要让后端真正使用 PostgreSQL + pgvector 时，使用 `compose.pgvector.yaml` override 启动（见下文）。

## pgvector override 启动方式

基础 `compose.yaml` 的 backend 不依赖 `db`，memory 模式无需数据库。要切换到 PostgreSQL + pgvector 存储后端：

```powershell
# 1. 启动数据库（仅 Compose 网络内可见，不映射宿主机端口）
docker compose up -d db

# 2. 构建 backend 镜像并显式执行迁移（应用启动不会自动迁移）
docker compose build backend
docker compose run --rm backend alembic upgrade head

# 3. 用 pgvector override 启动 backend 和 frontend
docker compose -f compose.yaml -f compose.pgvector.yaml up -d backend frontend
```

PowerShell 单行等价命令：

```powershell
docker compose -f compose.yaml -f compose.pgvector.yaml up -d backend frontend
```

`compose.pgvector.yaml` 只做两件事：把 `VECTOR_STORE_BACKEND` 设为 `pgvector`，并让 backend `depends_on db (condition: service_healthy)`，等待数据库健康后再启动。数据库未就绪或 schema 未迁移时后端进入降级状态：`/health` 返回 `degraded`，存储相关接口返回稳定的 503（详见 [database.md](database.md)）。

backend 重启方式（保留数据）：

```powershell
docker compose -f compose.yaml -f compose.pgvector.yaml restart backend
```

`postgres-data` 卷（`course-rag-postgres-data`）保存 PostgreSQL 数据目录：普通停止、重启和重新创建容器都不会删除它，pgvector 模式重启后文档、Chunk 与 Embedding 全部复用，不会重新计算。

切回 memory：

```powershell
docker compose up -d backend frontend
```

memory 模式不依赖数据库，`db` 服务可以停止或删除而不影响后端。

## 前置条件

- Docker Desktop 或 Docker Engine 已启动
- Docker Compose v2 可用
- 首次启动可以访问 Hugging Face，以下载默认 Embedding 模型
- 建议为镜像、Python 依赖和模型缓存预留数 GB 磁盘空间

检查命令：

```powershell
docker --version
docker compose version
docker info
```

## 配置

可先复制环境模板：

```powershell
Copy-Item .env.example .env
```

`.env` 可以配置：

```env
APP_PORT=8080
LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
LLM_TIMEOUT_SECONDS=30
RAG_MIN_RELEVANCE_SCORE=0.35
MAX_UPLOAD_BYTES=10485760
RAG_RERANKER_ENABLED=false
RAG_RERANKER_MODEL=
RAG_RERANKER_CANDIDATE_TOP_K=15
VECTOR_STORE_BACKEND=memory
POSTGRES_DB=course_rag
POSTGRES_USER=course_rag
POSTGRES_PASSWORD=course_rag
DATABASE_URL=postgresql+psycopg://course_rag:course_rag@db:5432/course_rag
```

`LLM_API_KEY` 和 `LLM_MODEL` 为空时，容器仍可完成健康检查、语义检索和文档管理；只有智能问答返回现有的 LLM 未配置提示。

数据库变量是基础设施默认值：`VECTOR_STORE_BACKEND=memory` 时后端不会连接数据库，`DATABASE_URL` 只在显式执行 `alembic` 迁移（或未来切换 `pgvector` 后端）时使用。这些是本地开发值，生产环境必须替换密码。

Compose 为 RAG 相关设置提供与应用程序内建默认一致的安全默认值：Reranker 默认关闭、相关性阈值默认 `0.35`、Reranker 候选池默认 `15`：

```env
RAG_RERANKER_ENABLED=false
RAG_RERANKER_MODEL=
RAG_RERANKER_CANDIDATE_TOP_K=15
RAG_MIN_RELEVANCE_SCORE=0.35
```

这些默认值不是永久固定的：用户可以通过本地 `.env` 或进程环境变量显式覆盖，Compose 会把覆盖后的值原样透传给后端。未配置时容器行为与直接运行 `uvicorn src.api:app` 完全一致（Reranker 关闭、相关性阈值 0.35）。`scripts/docker_smoke_test.py` 在隔离环境下运行，校验容器实际收到的就是这些安全默认值；默认 Smoke Test 不启用 Reranker，Docker 验证流程也不会下载 Reranker 模型。首次启动后端仍可能下载默认 Embedding 模型。`MAX_UPLOAD_BYTES` 默认仍为 `10485760`。Nginx 的请求体上限略高于 10 MiB，用来容纳 multipart 头部；实际文件大小仍由 FastAPI 按 `MAX_UPLOAD_BYTES` 校验。

`.env` 可能包含密钥，已被 Git 和 Docker 构建上下文排除。不要提交或分享 `docker compose config` 的完整输出，因为其中可能包含展开后的环境变量。

## 一键启动

在仓库根目录执行：

```powershell
docker compose up --build -d
```

首次构建会安装 CPU 版 PyTorch 和 Python 依赖；首次启动后端还会下载默认 Hugging Face 模型，因此耗时明显长于后续启动。观察进度：

```powershell
docker compose logs -f backend
```

查看容器和健康状态：

```powershell
docker compose ps
```

启动完成后：

- Web 应用：`http://127.0.0.1:8080`
- 经 Nginx 访问 FastAPI 健康接口：`http://127.0.0.1:8080/api/health`
- Nginx 自身健康接口：`http://127.0.0.1:8080/nginx-health`

PowerShell 验证：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/api/health
Invoke-WebRequest http://127.0.0.1:8080/nginx-health
```

如果 `8080` 已占用，在 `.env` 设置其他端口，例如 `APP_PORT=18080`，然后访问 `http://127.0.0.1:18080`。

## 持久化

Compose 使用三个具名卷：

| 卷 | 容器挂载点 | 保存内容 |
| --- | --- | --- |
| `course-rag-runtime` | `/app/data/runtime` | `uploads/` 上传原文件和 `documents.json` 元数据（memory 模式） |
| `course-rag-huggingface-cache` | `/cache/huggingface` | 默认 Embedding 模型等 Hugging Face 缓存 |
| `course-rag-postgres-data` | `/var/lib/postgresql/data` | PostgreSQL 数据目录（pgvector 模式保存文档、Chunk 与 Embedding） |

查看卷：

```powershell
docker volume inspect course-rag-runtime
docker volume inspect course-rag-huggingface-cache
docker volume inspect course-rag-postgres-data
```

普通停止、重启和重新创建容器不会删除具名卷：

```powershell
docker compose restart
docker compose down
docker compose up -d
```

`docker compose down` 只删除容器和网络，上传文档、元数据、模型缓存和数据库数据仍在。以下命令会永久删除本项目的三个具名卷，只有明确要清空所有数据时才执行：

```powershell
docker compose down --volumes
```

## 数据库迁移

`db` 服务默认不映射宿主机端口，只在 Compose 网络内可见。启动并迁移数据库：

```powershell
docker compose up -d db
docker compose build backend          # 镜像需包含新依赖与迁移文件
docker compose run --rm backend alembic upgrade head
docker compose run --rm backend alembic current
```

回滚与重新应用：

```powershell
docker compose run --rm backend alembic downgrade base
docker compose run --rm backend alembic upgrade head
```

迁移只创建项目表；`vector` 扩展在 downgrade 时保留，完整清理需管理员显式执行 `DROP EXTENSION vector`。应用启动不会自动执行迁移。详见 [database.md](database.md)。

## 完整启动与重建持久化测试

仓库提供了一个只依赖 Python 标准库和 Docker CLI 的测试脚本：

```powershell
python scripts/docker_smoke_test.py
```

脚本会依次：

1. 用临时环境变量执行 `docker compose config`，验证 Reranker/阈值等设置可以被显式覆盖（只展开配置，不启动容器，也不会加载 Reranker 模型）。
2. 构建并启动两个容器，等待 Nginx 和 FastAPI 可用。
3. 验证 Vue 静态页面、两个健康接口，以及容器实际收到的 RAG/LLM 设置是安全默认值；确认健康接口不泄露任何密钥。
4. 通过 Nginx 上传一份带唯一随机令牌的临时 TXT 文档（文件名和内容都包含该令牌，避免与历史残留数据混淆）。
5. 确认 Hugging Face 缓存非空并写入唯一卷标记。
6. 执行 `docker compose down` 和 `docker compose up -d`，真正重新创建容器但保留具名卷。
7. 验证上传文档及其元数据仍存在、启动时重建的索引能检索到该文档（按令牌检索）、模型缓存标记仍存在。
8. 删除临时文档和测试标记，并验证元数据、上传文件、缓存标记都已真正清除，服务继续运行。

脚本与本地环境隔离：每次 Compose 调用都会使用一个临时生成的环境文件（只含安全默认值）并通过净化后的进程环境运行，因此即使本地根目录 `.env` 或当前 shell 导出了 `LLM_*` / `RAG_*` 变量，测试启动的容器也只会收到 `RAG_RERANKER_ENABLED=false`、`RAG_RERANKER_CANDIDATE_TOP_K=15`、`RAG_MIN_RELEVANCE_SCORE=0.35` 和空的 LLM 配置——绝不会启用真实 Reranker 或携带真实 LLM 凭据。脚本校验的也正是这些默认值，并额外验证后端容器以非 root 用户运行、镜像内没有复制根目录 `.env`、容器环境变量中没有非空的 LLM/RAG 值；临时环境文件在测试结束后删除。

默认等待后端最多 900 秒。模型已经构建且只想复用镜像时：

```powershell
python scripts/docker_smoke_test.py --skip-build
```

更换端口后只需传入新的地址，脚本会从 `--base-url` 自动解析端口并写入隔离环境文件，容器映射与测试访问的端口保持一致：

```powershell
python scripts/docker_smoke_test.py --base-url http://127.0.0.1:18080
```

测试失败时脚本会保留当时的容器和临时数据，方便查看日志；修复或确认原因后可以在页面删除测试文档，或使用文档 API 删除。

## 常用运维命令

```powershell
# 查看服务状态
docker compose ps

# 跟踪全部日志
docker compose logs -f

# 只看后端最近 200 行
docker compose logs --tail 200 backend

# 重启服务（保留卷）
docker compose restart

# 修改代码后重建并启动
docker compose up --build -d

# 停止并删除容器和网络（保留卷）
docker compose down
```

## 排障

后端长时间处于 `starting`：

- 第一次启动通常正在下载并加载 Embedding 模型。
- 使用 `docker compose logs -f backend` 查看下载或网络错误。
- 确认 Docker 可以访问 Hugging Face，且磁盘空间充足。
- 后续启动会复用 `course-rag-huggingface-cache`，不应重复完整下载。

前端健康但 `/api/health` 返回 502：

- Nginx 已启动，但 FastAPI 仍在加载模型或启动失败。
- 查看 `docker compose ps` 和 `docker compose logs backend`。
- Compose 健康检查为模型首次下载预留了 5 分钟启动宽限期；慢速网络下测试脚本会继续等待最多 15 分钟。

上传返回 413：

- FastAPI 默认只接受不超过 10 MiB 的文件。
- Nginx 已为 multipart 开销预留空间，最终限制仍由 `MAX_UPLOAD_BYTES` 决定。
- 如果提高 `MAX_UPLOAD_BYTES` 到超过 12 MiB，还需要同步调整 `frontend/nginx.conf` 的 `client_max_body_size` 并重建前端镜像。

端口冲突：

- 在 `.env` 修改 `APP_PORT`，然后重新执行 `docker compose up -d`。

## 当前部署边界

这是本地演示和后续单机云部署的基础方案，不包含 TLS、身份认证、对象存储、多副本共享存储、GPU、Kubernetes 或自动化发布。后端不向宿主机发布端口、容器以非 root 用户运行并启用 `no-new-privileges`，密钥只通过 `.env` 注入、不写入镜像。具名卷属于当前 Docker 主机；迁移到另一台主机前需要单独备份。云部署阶段还应在外层补充 HTTPS、密钥管理和访问控制。

数据库服务同样是基础设施边界：默认 `memory` 后端不使用数据库，`db` 不映射宿主机端口、不带密码之外的认证加固、不做备份。`pgvector` 后端已可用，但仅在显式使用 `compose.pgvector.yaml` 时接管存储；本方案不提供生产级多租户隔离、高并发保证、水平扩展、ANN 索引、云对象存储或自动数据迁移。数据库 readiness 由 `compose.pgvector.yaml` 的 `depends_on` 与后端自身的 schema 检查共同保证。
