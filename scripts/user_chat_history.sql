-- =====================================================================
-- rag-psychology 长期记忆:user_chat_history 建表 + 相似历史检索函数
-- 执行方式:psql -f scripts/user_chat_history.sql(幂等,可重复执行)
-- 检索维度:按 user_id 全量检索该用户所有会话的相似历史(不按会话隔离)
-- 维度说明:.env EMBEDDING_MODEL=text-embedding-v3,输出 1024 维,
-- 与 config/settings.py 的 VECTOR_DIMENSION 保持一致;若切 text-embedding-v4
-- 并输出 2048 维,需同步改此处 VECTOR(1024) → VECTOR(2048) 后重建索引。
--
-- 双向量说明(2026-09-01 新增):
-- - embedding:     仅 query 的向量(兼容存量数据,保留)
-- - qa_embedding:  query + answer 拼接后的向量(检索主用)
--   匹配语义从「问题↔问题」升级为「问题↔问答内容」:用户换措辞
--   ("睡眠不好" → "老是失眠")时,靠历史 answer 的语义仍能召回。
--   检索函数优先用 qa_embedding,存量行(该列为 NULL)自动回退 embedding,
--   无需回填即可平滑升级;新写入的行由 modules/memory.py 同时打两列。
-- =====================================================================

CREATE TABLE IF NOT EXISTS user_chat_history (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(100) NOT NULL,
    query TEXT NOT NULL,
    answer TEXT,
    embedding VECTOR(1024),
    qa_embedding VECTOR(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- 存量库幂等升级:补加 qa_embedding 列(已存在则跳过)
ALTER TABLE user_chat_history ADD COLUMN IF NOT EXISTS qa_embedding VECTOR(1024);

-- 按用户过滤检索
CREATE INDEX IF NOT EXISTS idx_user_chat_history_user_id
    ON user_chat_history (user_id);

-- 向量近似检索(HNSW,余弦距离;数据量大时检索成本恒定)
CREATE INDEX IF NOT EXISTS idx_user_chat_history_embedding
    ON user_chat_history USING hnsw (embedding vector_cosine_ops);

-- qa 向量检索索引(qa_embedding 为 NULL 的行不进入该索引,由查询回退 embedding)
CREATE INDEX IF NOT EXISTS idx_user_chat_history_qa_embedding
    ON user_chat_history USING hnsw (qa_embedding vector_cosine_ops);

-- 相似历史检索函数:按用户过滤,余弦相似度 top5
-- p_query_vector: json 数组文本,如 "[0.1,0.2,...]"
-- p_user_id: 用户 ID;NULL 时不限用户
-- 检索列优先 qa_embedding(问答内容语义),NULL 时回退 embedding(query 语义)
CREATE OR REPLACE FUNCTION public.fn_search_chat_history(
    p_query_vector json,
    p_user_id      text DEFAULT NULL
)
 RETURNS TABLE(
    id bigint,
    user_id text,
    query text,
    answer text,
    created_at timestamp with time zone,
    cosine_similarity double precision
)
 LANGUAGE plpgsql
AS $function$
BEGIN
    RETURN QUERY
    SELECT
        h.id,
        h.user_id::text,
        h.query::text,
        h.answer::text,
        h.created_at,
        1 - (COALESCE(h.qa_embedding, h.embedding) <=> qv.v) AS cosine_similarity
    FROM user_chat_history h,
         (SELECT p_query_vector::text::vector AS v) qv
    WHERE (p_user_id IS NULL OR h.user_id = p_user_id)
      AND (h.qa_embedding IS NOT NULL OR h.embedding IS NOT NULL)
    ORDER BY COALESCE(h.qa_embedding, h.embedding) <=> qv.v
    LIMIT 5;
END;
$function$;
