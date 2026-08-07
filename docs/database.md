# PostgreSQL 与 pgvector 存储后端

本文档说明本仓库的两种存储后端：默认的 `memory`（JSON + 内存索引）与 `pgvector`（PostgreSQL + pgvector 向量检索）。`memory` 仍是默认后端；`pgvector` 后端已完整实现并接管上传、删除、检索与问答。

## 1. 架构

```text
浏览器 ── frontend（Nginx）── backend（FastAPI）
                              ├── memory（默认）:
                              │    检索: KnowledgeIndex（内存 NumPy 余弦相似度）
                              │    元数据: DocumentRepository（data/runtime/documents.json）
                              │    不连接 PostgreSQL
                              └── pgvector:
                                    documents 表（文档元数据）
                                    chunks 表（Chunk 文本、来源、VECTOR(384) Embedding）
                                    PostgreSQL 内执行 pgvector 余弦相似度检索
```

| 组件 | 作用 |
| --- | --- |
| `src/storage/protocols.py` | 应用层统一协议：`DocumentManagerProtocol` / `ChunkRetrieverProtocol` |
| `src/storage/runtime.py` | `StorageRuntime` 装配：按 `VECTOR_STORE_BACKEND` 选择 memory 或 pgvector |
| `src/storage/pgvector_store.py` | PostgreSQL 持久化与 pgvector 检索（事务、向量验证、异常脱敏） |
| `src/pgvector_ingestion_service.py` | pgvector 运行时：上传校验、文件存储、切分、Embedding、删除协调 |
| `src/database/` | 配置解析、engine / session、ORM 模型、schema 检查 |
| `migrations/` | Alembic 迁移环境与 schema 版本 |

## 2. 选择后端

```env
VECTOR_STORE_BACKEND=memory      # 默认：现有 JSON + 内存索引，不连接数据库
VECTOR_STORE_BACKEND=pgvector    # PostgreSQL + pgvector，要求有效 DATABASE_URL
```

- 未设置、空值或 `memory`（大小写不敏感）→ 内存后端，**不读取** `DATABASE_URL`、不创建 engine、不连接 PostgreSQL
- `pgvector` → 要求有效的 `postgresql+psycopg://` 连接串；缺失或非法时应用进入降级状态（见第 7 节）
- 任何其他值都会抛出稳定的 `DatabaseConfigurationError`（“数据库配置无效”），错误消息不包含连接串或密码

## 3. memory 模式启动

```powershell
uvicorn src.api:app --reload
```

memory 模式：

- 上传文档写入 `data/runtime/uploads/`，元数据写入 `data/runtime/documents.json`
- 检索使用 `KnowledgeIndex` 与 NumPy 余弦相似度
- 应用启动不创建数据库 engine、不连接 PostgreSQL、不执行 Alembic
- 即使 `db` 服务停止，memory 模式后端照常工作
- `documents.json` 中的数据保持原样，不会被迁移到 PostgreSQL

## 4. pgvector 模式启动

先启动数据库并执行迁移（数据库只在 Compose 网络内暴露，不映射宿主机端口）：

```powershell
docker compose up -d db
docker compose build backend
docker compose run --rm backend alembic upgrade head
docker compose run --rm backend alembic current
```

再以 pgvector 模式启动应用：

```powershell
docker compose -f compose.yaml -f compose.pgvector.yaml up -d backend frontend
```

也可以在本地直接运行（需要本机可访问的 PostgreSQL）：

```powershell
$env:VECTOR_STORE_BACKEND="pgvector"
$env:DATABASE_URL="postgresql+psycopg://course_rag:course_rag@localhost:5432/course_rag"
uvicorn src.api:app --reload
```

pgvector 模式启动时的初始化顺序：

1. 解析 `DATABASE_URL`（缺失/非法 → 降级）
2. 创建 SQLAlchemy engine 与 session factory
3. 检查数据库可连接（`SELECT 1`）
4. 检查 schema 就绪（Alembic revision、`vector` 扩展、`documents` / `chunks` 表）
5. 创建 `PgVectorStore` 与 `PgVectorIngestionService`
6. 初始化内置文档 `builtin-knowledge`（首次写入，重启复用）
7. 清理遗留 tombstone 文件（best-effort）

**应用启动不会自动执行 Alembic 迁移**；schema 变更始终是显式运维动作。

## 5. 迁移命令

```powershell
# 查看当前状态
docker compose run --rm backend alembic current
docker compose run --rm backend alembic history

# 升级到最新
docker compose run --rm backend alembic upgrade head

# 回滚后重新应用（用于验证迁移链）
docker compose run --rm backend alembic downgrade base
docker compose run --rm backend alembic upgrade head
```

迁移内容：

1. `CREATE EXTENSION IF NOT EXISTS vector`
2. 创建 `documents` 表（非负与 status 检查约束）
3. 创建 `chunks` 表（`VECTOR(384)` embedding 列、外键 `ON DELETE CASCADE`、唯一约束与检查约束）
4. 创建普通 B-tree 索引 `ix_chunks_document_id`（**没有** HNSW / IVFFlat）

## 6. compose.pgvector.yaml

`compose.pgvector.yaml` 是 pgvector 模式的 Compose override：

- 把 `VECTOR_STORE_BACKEND` 设为 `pgvector`
- 设置 `DATABASE_URL` 指向 Compose 网络内的 `db:5432`
- 给 backend 增加 `depends_on: db (condition: service_healthy)`，等待数据库就绪

基础 `compose.yaml` 保持 `VECTOR_STORE_BACKEND` 默认 `memory`，backend **不**依赖 `db` 服务，memory 模式启动时不需要数据库。

## 7. 数据库不可用时的行为（降级）

当选择 `VECTOR_STORE_BACKEND=pgvector` 但出现：

- `DATABASE_URL` 缺失或非法
- 数据库不可连接
- Alembic 未执行 / revision 不匹配 / `vector` 扩展缺失 / 表缺失

应用**仍能启动**并响应：

```text
GET /health  → 200，status="degraded"，retrieval_ready=false，rag_ready=false，chunk_count=0
POST /search → 503 STORAGE_*（配置无效 / 不可用 / schema 未就绪）
POST /ask    → 503 STORAGE_*
GET  /documents → 503 STORAGE_*
POST /documents → 503 STORAGE_*
DELETE /documents/{id} → 503 STORAGE_*
```

错误码映射：

| 异常 | HTTP | code | 消息 |
| --- | --- | --- | --- |
| `DatabaseConfigurationError` | 503 | `STORAGE_NOT_CONFIGURED` | 存储服务配置无效 |
| `DatabaseConnectionError` | 503 | `STORAGE_UNAVAILABLE` | 存储服务暂时不可用 |
| `DatabaseSchemaError` | 503 | `STORAGE_SCHEMA_NOT_READY` | 数据库结构尚未准备完成 |
| `DatabaseOperationError` | 503 | `STORAGE_UNAVAILABLE` | 存储服务暂时不可用 |

响应绝不会包含 `DATABASE_URL`、用户名、密码、主机名、SQL 或驱动错误文本。memory 模式的数据库故障不影响应用。

## 8. 查看数据

```powershell
docker compose exec -T db psql -U course_rag -d course_rag -c "SELECT count(*) FROM documents;"
docker compose exec -T db psql -U course_rag -d course_rag -c "SELECT count(*) FROM chunks;"
docker compose exec -T db psql -U course_rag -d course_rag -c "\d documents"
docker compose exec -T db psql -U course_rag -d course_rag -c "\d chunks"
```

查看 vector 列与扩展版本：

```powershell
docker compose exec -T db psql -U course_rag -d course_rag `
  -c "SELECT document_id, chunk_index, vector_dims(embedding) AS dims FROM chunks LIMIT 5;"
docker compose exec -T db psql -U course_rag -d course_rag `
  -c "SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';"
```

查看 Alembic revision：

```powershell
docker compose run --rm backend alembic current
```

## 9. 删除级联验证

删除文档后，其 Chunk 通过外键 `ON DELETE CASCADE` 自动删除：

```powershell
docker compose exec -T db psql -U course_rag -d course_rag `
  -c "SELECT count(*) FROM chunks c JOIN documents d ON d.document_id = c.document_id WHERE d.document_id = '<文档ID>';"
```

删除前应为非 0，删除后该查询返回 0。

## 10. 重启持久化验证

```powershell
# 上传一份文档后重启 backend
docker compose -f compose.yaml -f compose.pgvector.yaml restart backend
# 文档、Chunk 与 Embedding 都保存在 postgres-data 卷中，重启后直接复用
docker compose -f compose.yaml -f compose.pgvector.yaml exec -T db psql -U course_rag -d course_rag -c "SELECT count(*) FROM documents;"
```

应用重启**不会**重新计算已上传文档和内置文档的 Embedding：数据库里已保存的向量直接复用。

## 11. 如何切回 memory

```powershell
docker compose -f compose.yaml up -d backend frontend
```

或不带 override 启动。memory 模式继续使用 `data/runtime/documents.json`（其中已有的上传记录原样可用）；pgvector 模式写入的 PostgreSQL 数据不会被读取，两个数据集彼此独立。

## 12. 当前不自动迁移 documents.json

`documents.json` 中的现有数据**不会**被自动导入 PostgreSQL。数据库表保持只有新上传内容的状态，直到后续显式实现数据迁移逻辑。不要在 pgvector 模式下期待历史 JSON 记录自动出现。

## 13. 当前不自动同步 knowledge.txt 变化

内置文档 `knowledge.txt` 只在首次写入时切分并计算 Embedding；应用重启时若内置文档已存在则直接复用，**不检测内容变化**。如果 `knowledge.txt` 内容发生变化，需要后续显式重建机制；本仓库当前不提供自动增量同步。

## 14. 测试数据清理

安全清理测试期间写入的上传文档（保留内置文档）：

```powershell
docker compose exec -T db psql -U course_rag -d course_rag `
  -c "DELETE FROM documents WHERE document_id <> 'builtin-knowledge';"
```

完全清空数据库（会删除全部数据，包括内置文档与历史上传）：

```powershell
docker compose exec -T db psql -U course_rag -d course_rag -c "DELETE FROM chunks; DELETE FROM documents;"
```

**不要随意执行 `docker compose down -v`**：它会永久删除 `postgres-data`、`rag-runtime` 与 `huggingface-cache` 三个具名卷，包括上传原文件、模型缓存和全部数据库数据。

## 15. 常见错误

| 现象 | 原因与处理 |
| --- | --- |
| `/health` 返回 `degraded` | pgvector 配置或数据库未就绪；查看第 7 节错误码 |
| `alembic upgrade head` 报数据库配置无效 | `DATABASE_URL` 未设置、为空或不是 `postgresql+psycopg://` 开头 |
| `db` 容器一直 unhealthy | 检查 `docker compose logs db`；本地首次启动需要几秒完成初始化 |
| 端口冲突 | `db` 不映射宿主机端口，不存在冲突；宿主机调试请用 override 临时映射 |
| 重复执行 migration 报表已存在 | 先 `alembic current` 确认版本，正常情况不会重复建表 |
| 上传后 /search 检索不到 | 确认查询文本与上传 Chunk 语义相近；pgvector 只检索 `status='ready'` 的文档 |

## 当前部署边界

- 数据库仅本地开发用途，未配置 TLS、备份、多副本或用户权限体系
- 本实现不提供生产级多租户隔离、高并发保证、水平扩展或 ANN 索引
- 不支持云对象存储、自动数据迁移或 knowledge.txt 自动同步
- 生产级数据库运维（高可用、备份、监控）不在本仓库范围
