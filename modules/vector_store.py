"""
向量存储和检索模块
基于Chroma实现心理学知识的存储和检索
"""
from typing import List, Optional, Dict
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from config.settings import settings


class PsychologyVectorStore:
    """心理学向量存储"""

    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = None,
    ):
        self.collection_name = collection_name or settings.COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR

        # 初始化OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model or settings.EMBEDDING_MODEL,
            openai_api_key=settings.OPENAI_API_KEY,
            openai_api_base=settings.OPENAI_API_BASE,
            tiktoken_enabled=False,
            check_embedding_ctx_length=False,
            chunk_size=10,
        )

        # 初始化Chroma向量存储
        self.vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """添加或更新文档到向量存储。"""
        if not documents:
            return []

        stored_ids = self.vectorstore.add_documents(documents, ids=ids)
        print(f"成功写入 {len(stored_ids)} 个文档到向量存储")
        return stored_ids

    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """语义相似度搜索"""
        k = k or settings.RETRIEVAL_TOP_K
        return self.vectorstore.similarity_search(query, k=k, filter=filter_dict)

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = None,
        filter_dict: Optional[Dict] = None,
    ) -> List[tuple]:
        """带相关性分数的语义搜索（分数越高越相关）"""
        k = k or settings.RETRIEVAL_TOP_K
        return self.vectorstore.similarity_search_with_relevance_scores(
            query, k=k, filter=filter_dict
        )

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter_dict: Optional[Dict] = None,
    ) -> List[Document]:
        """最大边际相关性搜索（平衡相关性和多样性）"""
        k = k or settings.RETRIEVAL_TOP_K
        return self.vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter_dict,
        )

    def delete_collection(self):
        """清空并重新创建当前集合。"""
        self.vectorstore.reset_collection()
        print(f"已重建集合: {self.collection_name}")

    def get_collection_stats(self) -> Dict:
        """获取集合统计信息"""
        count = self.vectorstore._collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
        }

    def as_retriever(self, k: int = None, search_type: str = "similarity"):
        """返回LangChain retriever对象"""
        k = k or settings.RETRIEVAL_TOP_K

        if search_type == "mmr":
            return self.vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": 20,
                    "lambda_mult": 0.5,
                }
            )
        else:
            return self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
