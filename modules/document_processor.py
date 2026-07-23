"""
文档处理模块
支持PDF、TXT、Markdown等格式的心理学文献处理
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config.settings import settings


class DocumentProcessor:
    """心理学文档处理器"""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        )

    def load_document(self, file_path: str) -> List[Document]:
        """加载单个文档"""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()

        if suffix == ".pdf":
            loader = PyPDFLoader(str(path))
        elif suffix == ".txt":
            loader = TextLoader(str(path), encoding="utf-8")
        elif suffix in [".md", ".markdown"]:
            loader = UnstructuredMarkdownLoader(str(path))
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(path))
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")

        return loader.load()

    def load_directory(self, dir_path: str, recursive: bool = True) -> List[Document]:
        """加载目录下所有支持的文档"""
        path = Path(dir_path)
        if not path.exists():
            raise FileNotFoundError(f"目录不存在: {dir_path}")

        documents = []
        supported_extensions = {".pdf", ".txt", ".md", ".docx"}

        pattern = "**/*" if recursive else "*"
        for file_path in path.glob(pattern):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    docs = self.load_document(str(file_path))
                    # 添加来源元数据
                    for doc in docs:
                        doc.metadata["source"] = str(file_path)
                        doc.metadata["filename"] = file_path.name
                    documents.extend(docs)
                except Exception as e:
                    print(f"加载文件失败 {file_path}: {e}")

        return documents

    def split_documents(
        self,
        documents: List[Document],
        add_metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """分割文档并添加元数据"""
        chunks = self.text_splitter.split_documents(documents)

        # 添加额外元数据
        if add_metadata:
            for chunk in chunks:
                chunk.metadata.update(add_metadata)

        # 为每个chunk添加唯一ID
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{idx}"
            chunk.metadata["chunk_index"] = idx

        return chunks

    def process_and_split(
        self,
        source: str,
        is_directory: bool = True,
        metadata: Optional[Dict] = None,
    ) -> List[Document]:
        """处理并分割文档的统一接口"""
        if is_directory:
            documents = self.load_directory(source)
        else:
            documents = self.load_document(source)

        return self.split_documents(documents, add_metadata=metadata)

    def export_chunks(self, chunks: List[Document], output_path: str):
        """导出分块结果为JSON（用于调试）"""
        data = []
        for chunk in chunks:
            data.append({
                "chunk_id": chunk.metadata.get("chunk_id"),
                "content": chunk.page_content,
                "metadata": chunk.metadata,
            })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"已导出 {len(data)} 个chunks到 {output_path}")
