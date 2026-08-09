# Production single-host deployment

本文档把 course-rag 部署到一台 Linux VPS，通过域名和自动 HTTPS 对外提供安全访问。整个应用（PostgreSQL + pgvector、FastAPI backend、Vue frontend、Caddy 反向代理）运行在同一台主机上的 Docker Compose 里，Caddy 是唯一的公网入口。

阅读本文档的人需要具备基础的 Linux 与 Docker 知识，不需要阅读源码。

## Architecture

```text
Internet
  │
  ▼
Caddy :80/:443（HTTP → HTTPS）
  ├── /api/*  → FastAPI backend :8000（Caddy 剥掉 /api 前缀）
  │                         │
  │                         ▼
  │               PostgreSQL 17 + pgvector
  └── /*      → Vue / Nginx :80
```

所有服务运行在同一个 Compose 网络（`course-rag_default`）内，容器之间通过服务名互相访问。浏览器只与 Caddy 通信，永远不会直接连接 backend、frontend 或数据库。

## Requirements

- Ubuntu 24.04 LTS（或其他现代 Linux 发行版）
- 2 vCPU 起步，建议 4 vCPU
- 至少 4 GB RAM（默认 Embedding 模型需要一定内存；**4 GB 并不保证一定够用**，启用 Reranker 或换用更大模型时需求会显著上升）
- 磁盘容量建议预留 20 GB 以上：镜像（PyTorch 等）、Python 依赖、Embedding 模型缓存、PostgreSQL 数据、上传文档都落在本机
- Docker Engine（20.10+）
- Docker Compose plugin（v2.24+，本方案使用 `ports: !reset` 语法）

Docker Compose 版本检查：

```bash
docker --version
docker compose version
```

默认 Reranker 关闭（`RAG_RERANKER_ENABLED=false`）时资源占用更低；本部署文档不启用 Reranker。

## DNS

把域名解析到 VPS 的公网 IPv4：

- A 记录：`<DOMAIN>` → VPS IPv4 地址
- 有 IPv6 时：AAAA 记录 → VPS IPv6 地址

等待 DNS 生效后用下面任一命令确认（`<DOMAIN>` 替换为你的域名）：

```bash
dig <DOMAIN> +short
nslookup <DOMAIN>
```

Caddy 首次启动会向 Let's Encrypt 申请证书，要求域名已经正确解析到本机、且 80/443 从公网可达。**DNS 未生效前不要开始**，否则证书申请会失败。

## Firewall

放行以下端口：

| 端口 | 协议 | 用途 |
| --- | --- | --- |
| 22 | tcp | SSH（先放行，避免把自己锁在门外） |
| 80 | tcp | HTTP（Caddy 自动跳转 HTTPS 与 ACME 验证） |
| 443 | tcp | HTTPS |
| 443 | udp | 可选，HTTP/3（云平台不支持 UDP 时可省略） |

**不得**放行 `5432`、`8000`、`8080`——这些端口在部署中不会被发布，也不应暴露。

UFW 示例：

```bash
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
# 可选：
sudo ufw allow 443/udp
# 先确认 SSH 规则已就位，再启用：
sudo ufw enable
sudo ufw status
```

## Clone

```bash
git clone https://github.com/bluesblue320-hue/course-rag.git
cd course-rag
```

## Production environment

```bash
cp .env.production.example .env.production
chmod 600 .env.production
```

编辑 `.env.production`，填写：

- `DOMAIN`：你的域名（例如 `rag.example.com`）
- `POSTGRES_PASSWORD`：数据库密码（强随机）
- `DATABASE_URL`：见下一节
- `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`：你的 LLM 服务商配置

`.env.production` 包含真实密钥，已被 Git 忽略（`.gitignore`），也绝不会进入 Docker 镜像。不要提交、不要分享。

## Database password

`POSTGRES_PASSWORD` 与 `DATABASE_URL` 中的密码必须一致。`DATABASE_URL` 格式：

```
postgresql+psycopg://course_rag:<密码>@db:5432/course_rag
```

如果密码包含 URL 特殊字符（`@`、`:`、`/`、`?`、`#`、`%` 等），`DATABASE_URL` 中的密码必须 URL 编码：

| 字符 | 编码 |
| --- | --- |
| `@` | `%40` |
| `:` | `%3A` |
| `/` | `%2F` |
| `?` | `%3F` |
| `#` | `%23` |
| `%` | `%25` |

例如密码是 `s3cr@t:pw`，则 `DATABASE_URL=postgresql+psycopg://course_rag:s3cr%40t%3Apw@db:5432/course_rag`。

## Start

首次部署（构建镜像、启动数据库并执行迁移、启动全部服务）：

```bash
docker compose \
  --env-file .env.production \
  -f compose.yaml \
  -f compose.pgvector.yaml \
  -f compose.prod.yaml \
  up -d --build
```

启动顺序由 Compose 依赖保证：

```text
db healthy
  ↓
migrate（alembic upgrade head，一次性）
  ↓
backend healthy
  ↓
frontend healthy
  ↓
caddy
```

`migrate` 是一次性迁移服务：首次 `up` 时它会自动执行 `alembic upgrade head`，成功后退出（exit 0）。**迁移失败时 backend 不会启动**，避免在 schema 未就绪的情况下运行。

Compose dependencies 负责首次部署的 `db healthy → migrate → backend` 顺序。除此之外，production backend 自身在**每次容器进程启动**时还有一层只读 readiness gate：

1. 等待 PostgreSQL 可连接；
2. 使用现有 schema 检查确认 Alembic revision、pgvector extension、`documents` 与 `chunks` 均已就绪；
3. 检查通过后才以 `exec` 启动 Uvicorn。

门禁不会执行 migration，也不会加载 Embedding 或调用 LLM。它最多尝试 60 次、重试间隔 2 秒；达到上限仍未就绪时进程以非零状态退出，由 `restart: unless-stopped` 重新尝试。这样 Docker daemon 重启、VPS reboot 或手工重启 backend 时，即使 Compose dependency orchestration 没有重新执行，FastAPI 也不会因抢先连接数据库而长期停留在 degraded storage 状态。

首次启动会下载默认 Embedding 模型（写入 `huggingface-cache` 卷），耗时明显长于后续启动，属正常现象：

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 -f backend
```

## Status

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  ps --all
```

预期状态：

```text
NAME                STATUS
course-rag-db       healthy
course-rag-migrate  exited (0)      ← 一次性迁移已成功完成
course-rag-backend  healthy
course-rag-frontend healthy
course-rag-caddy    running
```

## Health readiness

Local/base Compose 的 backend healthcheck 只验证 `/health` 的 HTTP 可达性。Production override 使用更严格的 semantic readiness probe，只有以下条件同时成立才判定容器 healthy：

- `status == "ok"`
- `retrieval_ready == true`
- `generation_ready == true`
- `rag_ready == true`

因此，即使 degraded `/health` 仍返回 HTTP 200，也不会让 production backend 被 Docker 标记为 healthy，frontend/Caddy 的首次 Compose 启动依赖也不会提前通过。

该探针只确认本地 production RAG runtime 已正确初始化；它**不会**实时探测远程 LLM provider 的网络可达性或服务状态。

## HTTPS

证书由 Caddy 自动申请并续期，无需手动操作。验证：

```bash
curl -I https://<DOMAIN>/
curl -fsS https://<DOMAIN>/api/health
```

第一个命令应返回 `HTTP/2 200`；第二个命令应返回 JSON（`status: "ok"`）。

如果证书申请失败，按 [Troubleshooting](#troubleshooting) 中 “Caddy cannot issue certificate” 一节排查。首次启动证书签发要求 DNS 已正确解析、80/443 从公网可达。

## Functional smoke test

浏览器打开 `https://<DOMAIN>`，测试：

- 页面正常打开
- 文档列表加载
- 上传一份文档
- 搜索能检索到内容
- Ask 提问能得到带引用的回答（citations）
- 删除刚上传的文档（可选）

## Logs

日志有界（每个服务 json-file 驱动，单文件最大 10 MB、保留 3 个文件），按需查看即可：

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 backend
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 caddy
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 db
```

不要使用 `docker logs` 不加限制地输出全部日志。

## Restart

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  restart db backend frontend caddy
```

重启后验证：文档列表仍显示已有文档、搜索仍可用。数据保存在具名卷中，重启不丢失。

## VPS reboot

直接重启 VPS：

```bash
sudo reboot
```

长期运行的 `db` / `backend` / `frontend` / `caddy` 由 `restart: unless-stopped` 在 Docker daemon 启动后自动恢复。

Docker daemon 直接恢复容器时不会重新执行 Compose 的 `depends_on` orchestration；production backend 因此在每次 process startup 都先运行 DB/schema readiness gate。PostgreSQL 启动较慢时，backend 会等待数据库可连接且 schema current，而不是直接启动为永久 degraded 的 FastAPI 进程。

启动等待期间，`https://<DOMAIN>/api/*` 可能短暂返回 502/503。数据库就绪后，backend 启动 Uvicorn，semantic healthcheck 通过，Caddy 的反向代理会自动恢复，无需修改路由或手工重启 Caddy。

重连后检查状态：

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  ps --all
```

确认 `db` / `backend` / `frontend` / `caddy` 自动恢复为 `healthy` / `running`。

## Persistence

| 具名卷 | 容器路径 | 保存内容 |
| --- | --- | --- |
| `course-rag-postgres-data` | `/var/lib/postgresql/data` | PostgreSQL 数据：文档元数据、Chunk、pgvector Embedding |
| `course-rag-runtime` | `/app/data/runtime` | 上传的原始源文档与运行时状态 |
| `course-rag-huggingface-cache` | `/cache/huggingface` | Hugging Face 模型缓存 |
| `course-rag-caddy-data` | `/data` | TLS 证书、ACME 状态 |
| `course-rag-caddy-config` | `/config` | Caddy 运行时状态 |

普通停止、重启、重建容器都不会删除具名卷。

## VERY IMPORTANT

**不要执行以下命令**，除非你明确要删除全部数据：

```bash
# 会永久删除 PostgreSQL 数据、上传文档、模型缓存、Caddy 证书：
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  down -v
```

在没有备份的生产机器上，也不要执行：

```bash
docker system prune --volumes
```

## Upgrade

升级到新版本（例如 `git pull` 拿到了包含新 Alembic migration 的代码）：

```bash
cd course-rag
git fetch
git pull --ff-only

# 1. 先构建包含新代码与新 migration 的 backend 镜像
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  build backend

# 2. 用刚构建的镜像显式执行数据库迁移（每次都会新建容器）
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  run --rm migrate

# 3. 重建前端并启动全部长期运行服务
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  up -d --build

# 4. 验证
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  ps --all
curl -fsS https://<DOMAIN>/api/health
```

说明：`migrate` 是 `restart: "no"` 的一次性服务。首次 `up` 时它自动执行一次；之后如果迁移容器已经成功退出，再次 `up` 不会可靠地重新执行它。因此**升级时统一使用 `run --rm migrate`**，它总是创建一个新容器执行 `alembic upgrade head`，结果可靠、可重复。

## Backup

备份必须同时覆盖数据库与上传的原始文档，只备份 PostgreSQL 是不够的。

先停止所有会写入数据的应用服务，再执行 PostgreSQL 逻辑备份与原始文件卷备份；这样两份备份共享同一个静止时间窗口：

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  stop caddy frontend backend
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  exec -T db \
  sh -c 'pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB"' \
  > course-rag.sql
docker run --rm -v course-rag-runtime:/data -v "$PWD":/backup \
  alpine tar czf /backup/course-rag-runtime.tar.gz -C /data .
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  up -d
```

把 `course-rag.sql` 与 `course-rag-runtime.tar.gz` 一起转移到安全位置。本仓库不提供自动备份任务。

## Restore

恢复 = 恢复 PostgreSQL + 恢复原始文件卷，两者必须来自同一次一致性备份。先停止 `caddy`、`frontend` 与 `backend`，并在任何覆盖前另存当前数据库和 `course-rag-runtime` 卷。随后由管理员把 `course-rag.sql` 导入一个已确认可覆盖的空目标数据库，并把 `course-rag-runtime.tar.gz` 解压到一个已确认为空的 runtime 卷；不要把归档直接叠加到仍含旧文件的卷。恢复完成后运行 `docker compose ... run --rm migrate`，再 `docker compose ... up -d`，最后检查 `/api/health`、文档列表与搜索结果。

数据库和 runtime 只恢复其中之一会造成元数据与原文件不一致。本仓库不提供全自动恢复脚本；破坏性覆盖前应让熟悉 PostgreSQL 和 Docker volume 的管理员复核目标卷名与备份时间点。

## Troubleshooting

### DNS not resolved

`dig <DOMAIN> +short` 返回空或错误 IP：

- 检查 DNS 记录是否已创建、TTL 是否已过期
- 检查 `DOMAIN` 在 `.env.production` 中是否拼写正确
- 证书签发前必须先修复 DNS

### Caddy cannot issue certificate

查看 Caddy 日志：

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 caddy
```

常见原因与处理：

- DNS 未生效 → 等 DNS 生效再重启 caddy
- 80/443 被防火墙/云安全组拦截 → 放行端口后 `docker compose ... restart caddy`
- 证书签发频率受限（Let's Encrypt rate limit）→ 等待后再试

### 80/443 blocked

- 云平台安全组 + VPS 本机 UFW 都要放行 80/443
- 用 `curl -I http://<DOMAIN>/` 确认 80 可达（应返回 308 跳转）
- Caddy 无法监听 80/443 时不会完成证书申请

### Backend keeps restarting

如果 backend 在 production 中持续重启，先查看有界日志：

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 backend
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 db
```

常见原因：

- PostgreSQL 仍在启动
- `DATABASE_URL` 错误或数据库不可达
- `POSTGRES_PASSWORD` 与 `DATABASE_URL` 中的密码不一致
- Alembic schema 不是当前 head
- pgvector extension 或所需表缺失

如果日志出现 `database readiness check failed after maximum attempts`，不要删除 healthcheck 或绕过 startup gate。应先修复数据库连接、凭据或 migration，再执行：

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  run --rm migrate
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  up -d backend
```

### backend unhealthy

```bash
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  ps
docker compose --env-file .env.production \
  -f compose.yaml -f compose.pgvector.yaml -f compose.prod.yaml \
  logs --tail=100 backend
```

- 首次启动在下载/加载 Embedding 模型，`/health` 有 5 分钟启动宽限期
- 磁盘空间不足会导致模型下载失败

### database connection failure

- 确认 `db` healthy（`docker compose ... ps`）
- 检查 `DATABASE_URL` 中的用户、密码、数据库名与 `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` 一致
- 密码含特殊字符时确认已 URL 编码

### Alembic migration failure

- 手动执行迁移查看错误：`docker compose ... run --rm migrate`
- 常见原因：`DATABASE_URL` 配置错误、数据库不可达
- 迁移失败时 backend 不会启动（`service_completed_successfully` 未满足），这是预期行为

### LLM configuration missing

- backend 日志出现 `LLM_NOT_CONFIGURED` / 503 → 检查 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` 是否已正确填写
- 三者都是 production 必需变量，`.env.production` 中缺失时 Compose 直接拒绝启动

### embedding model download slow/failure

- 首次启动从 Hugging Face 下载模型，网络慢时耐心等待
- 下载失败检查磁盘空间与网络；模型缓存在 `course-rag-huggingface-cache` 卷中，成功后重启不会重复下载

### disk full

```bash
df -h
docker system df
```

- 日志有界（10 MB × 3 文件/服务），不会无限增长
- PostgreSQL、上传文档、模型缓存是磁盘主要占用；按 [Backup](#backup) 定期转移备份后清理

### Docker volume missing

```bash
docker volume ls
docker volume inspect course-rag-postgres-data
```

- 五个具名卷必须存在：`course-rag-postgres-data`、`course-rag-runtime`、`course-rag-huggingface-cache`、`course-rag-caddy-data`、`course-rag-caddy-config`
- 卷丢失说明数据目录曾被删除（如 `docker compose down -v`），需要从备份恢复

## Public demo security warning

本应用**当前没有**：

- 用户认证（user authentication）
- 访问限流（rate limiting）
- 多用户隔离（multi-user isolation）
- 配额系统（quota system）

如果把实例直接长期暴露在公网，任何能访问 `https://<DOMAIN>` 的人都可能：

- 调用你的付费 LLM（消耗你的额度）
- 上传任意文档（消耗磁盘）
- 删除/修改共享知识库内容

因此**不要把带高额度付费 API Key 的实例长期无保护暴露公网**。

作为 Portfolio Demo / 演示用途，推荐：

- 使用低额度 / 带 spend-limit 的 API key（服务商侧设置消费上限）
- 演示期间短时开放，结束后关闭或移除公网访问
- 必要时在 VPS / CDN / 外层访问层加访问控制（如 IP 白名单、Basic Auth 反代）
- 定期检查磁盘占用与服务商 usage

本项目不自行实现 user auth，也不声称这是 multi-tenant production SaaS。本部署是 single-host production-style deployment，面向单用户/单团队演示与私有部署。
