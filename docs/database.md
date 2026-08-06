# PostgreSQL 与 pgvector 数据库基础设施

本文档说明本仓库提供的 PostgreSQL + pgvector 基础设施。**当前它只作为准备好的基础设施存在，不参与生产 RAG 请求**。

## 1. 当前架构

```text
浏览器 ── frontend（Nginx）── backend（FastAPI）
                              ├── 检索: KnowledgeIndex（内存 NumPy 余弦相似度）
                              ├── 元数据: DocumentRepository（data/runtime/documents.json）
                              └── 数据库（基础设施，默认不连接）
                                    db 服务（pgvector/pgvector:pg17，Compose 网络内）
```

| 组件 | 作用 |
| --- | --- |
| `src/database/config.py` | 解析 `VECTOR_STORE_BACKEND` 与 `DATABASE_URL`，校验非法值 |
| `src/database/engine.py` | 显式创建 SQLAlchemy engine / session factory，`SELECT 1` 连接检查 |
| `src/database/base.py` | Declarative Base 与 `DEFAULT_EMBEDDING_DIMENSION = 384` |
| `src/database/models.py` | `documents` 与 `chunks` ORM 模型 |
| `migrations/` | Alembic 迁移环境与首个 schema 版本 |

## 2. PR #10 只提供基础设施

本 PR 只建立数据库地基：依赖、配置、engine、ORM 模型、Docker 服务与首个 migration。它**不**切换任何现有 RAG 行为，`PgVectorStore` 由后续 PR 实现。

## 3. 默认 VECTOR_STORE_BACKEND=memory

```env
VECTOR_STORE_BACKEND=memory
```

- 未设置、空值或 `memory`（大小写不敏感）→ 内存后端，不要求 `DATABASE_URL`
- `pgvector` → 要求有效的 `postgresql+psycopg://` 连接串，否则启动配置校验会报错
- 任何其他值都会抛出稳定的 `DatabaseConfigurationError`（“数据库配置无效”），错误消息不包含连接串或密码

## 4. PostgreSQL 尚未接管 RAG 请求

memory 模式下：

- 上传文档仍写入 `data/runtime/uploads/`，元数据仍写入 `data/runtime/documents.json`
- 检索仍使用 `KnowledgeIndex` 与 NumPy 余弦相似度
- `/search`、`/ask`、`/documents` 的响应结构不变
- 应用启动不创建数据库 engine、不连接 PostgreSQL、不执行 Alembic
- 即使 `db` 服务停止，memory 模式后端照常工作

## 5. Docker 启动数据库

```powershell
docker compose up -d db
docker compose ps
docker compose logs db --tail 100
```

`db` 服务使用 `pgvector/pgvector:pg17`，数据库只通过 Compose 网络暴露（`expose: 5432`），**默认不映射到宿主机端口**。如果需要在宿主机上调试，可以临时用 override 暴露端口，例如：

```powershell
docker compose run --rm --service-ports db
```

不要在默认 `compose.yaml` 中打开 `5432:5432`。

## 6. Alembic upgrade

```powershell
docker compose build backend   # 镜像包含新依赖与迁移文件
docker compose run --rm backend alembic upgrade head
docker compose run --rm backend alembic current
docker compose run --rm backend alembic history
```

迁移内容：

1. `CREATE EXTENSION IF NOT EXISTS vector`
2. 创建 `documents` 表（含非负与 status 检查约束）
3. 创建 `chunks` 表（含 `VECTOR(384)` embedding 列、外键 CASCADE、唯一约束与检查约束）
4. 创建普通 B-tree 索引 `ix_chunks_document_id`

`DATABASE_URL` 必须存在且以 `postgresql+psycopg://` 开头；`migrations/env.py` 会覆盖 `alembic.ini` 中的占位值。

## 7. Alembic downgrade

```powershell
docker compose run --rm backend alembic downgrade base
```

- 删除 `chunks` 索引、`chunks` 表和 `documents` 表
- **不会**执行 `DROP EXTENSION vector`：扩展可能由数据库管理员或其他 schema 共用；如需彻底清理，由管理员显式执行 `DROP EXTENSION vector`
- downgrade 后再 `alembic upgrade head` 必须成功

## 8. 查看 vector 扩展

```powershell
docker compose exec -T db `
  psql -U course_rag -d course_rag `
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

## 9. 查看表结构

```powershell
docker compose exec -T db psql -U course_rag -d course_rag -c "\dt"
docker compose exec -T db psql -U course_rag -d course_rag -c "\d documents"
docker compose exec -T db psql -U course_rag -d course_rag -c "\d chunks"
```

## 10. named volume 持久化

`postgres-data` 卷（`course-rag-postgres-data`）保存数据库数据目录。普通停止、重启、重新创建容器都不会删除卷：

```powershell
docker compose stop db
docker compose start db
docker compose down        # 保留卷
```

只有 `docker compose down --volumes` 才会永久删除 `course-rag-postgres-data`（以及 `rag-runtime`、`huggingface-cache`）。

## 11. 数据库密码安全

- `.env.example` 中的 `POSTGRES_PASSWORD=course_rag` 只是本地开发默认值，**生产环境必须替换**
- `.env` 已被 `.gitignore` 排除，绝不能提交
- `DATABASE_URL` 错误消息、日志和异常均不包含密码、用户名、主机或完整连接串

## 12. 常见错误

| 现象 | 原因与处理 |
| --- | --- |
| `alembic upgrade head` 报数据库配置无效 | `DATABASE_URL` 未设置、为空或不是 `postgresql+psycopg://` 开头 |
| `db` 容器一直 unhealthy | 检查 `docker compose logs db`；本地首次启动需要几秒完成初始化 |
| 端口冲突 | `db` 不映射宿主机端口，不存在冲突；宿主机调试请用 override 临时映射 |
| 重复执行 migration 报表已存在 | 先 `alembic current` 确认版本，正常情况不会重复建表 |

## 13. 后续 PR #11 将实现 PgVectorStore

后续 PR 会用 `PgVectorStore` 把上传、删除、检索接入 PostgreSQL：写入 `documents` / `chunks`、按 `embedding <=>` 相似度检索、并按 `VECTOR_STORE_BACKEND` 切换后端。届时才会涉及数据库 readiness 与条件依赖。

## 14. 当前没有 HNSW / IVFFlat

本 PR 只建立普通 B-tree 索引。**没有**创建 HNSW、IVFFlat、cosine ANN 或部分向量索引；ANN 索引属于后续 PR 的显式决策。

## 15. 当前没有自动迁移 documents.json

`documents.json` 中的现有数据不会被自动导入 PostgreSQL；数据库表保持空表状态，直到后续 PR 显式实现数据迁移逻辑。

## 当前部署边界

- 数据库仅本地开发用途，未配置 TLS、备份、多副本或用户权限体系
- `VECTOR_STORE_BACKEND=pgvector` 尚未生效，即使开启也不会改变当前 RAG 行为
- 生产级数据库运维（高可用、备份、监控）不在本 PR 范围
