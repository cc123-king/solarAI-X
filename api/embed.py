import os
import sys


def _progress(*values, **options):
    print(*values, file=sys.stderr, **options)

import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
import glob
from tqdm import tqdm
from retrieval_utils import LexicalIndex, chunk_text, hybrid_rerank

try:
    from FlagEmbedding import FlagAutoModel
except ImportError:  # 让MCP服务可在未安装Embedding依赖时降级启动
    FlagAutoModel = None


MODELNAME='BAAI/bge-base-en-v1.5'
QUERYQUESTION="Represent this sentence for searching relevant passages:"

class TextEmbedding:
    """
    文本嵌入类，用于存储文本与embedding对，并提供相关功能
    """
    
    def __init__(self, model_name: str = MODELNAME, 
                 query_instruction: str = QUERYQUESTION,
                 use_fp16: bool = True):
        """
        初始化TextEmbedding类
        
        Args:
            model_name: 使用的模型名称
            query_instruction: 检索指令
            use_fp16: 是否使用fp16精度
        """
        requested_backend = os.getenv("RAG_BACKEND", "auto").strip().lower()
        if requested_backend not in {"auto", "semantic", "lexical"}:
            raise ValueError("RAG_BACKEND 必须是 auto、semantic 或 lexical")

        self.model = None
        self.backend = "lexical"
        self.degraded_reason: str | None = None
        if requested_backend != "lexical":
            if FlagAutoModel is None:
                if requested_backend == "semantic":
                    raise RuntimeError(
                        "RAG_BACKEND=semantic，但未安装 FlagEmbedding。"
                    )
                self.degraded_reason = (
                    "未安装 FlagEmbedding，已自动切换为本地关键词检索。"
                )
            else:
                try:
                    self.model = FlagAutoModel.from_finetuned(
                        model_name,
                        query_instruction_for_retrieval=query_instruction,
                        use_fp16=use_fp16,
                    )
                    self.backend = "semantic"
                except Exception as error:
                    if requested_backend == "semantic":
                        raise
                    self.degraded_reason = (
                        f"语义模型加载失败，已切换为关键词检索：{error}"
                    )

        self.text_embeddings: Dict[str, np.ndarray | None] = {}
        # 添加文件信息存储
        self.file_info: Dict[str, str] = {}  # 文本 -> 文件名映射
        self._lexical_index: LexicalIndex | None = None
        
    def add_texts(self, texts: List[str], truncate_length: Optional[int] = None, file_names: Optional[List[str]] = None) -> None:
        """
        批量添加文本并计算embedding
        
        Args:
            texts: 文本列表
            truncate_length: 截断长度，如果为None则不截断
            file_names: 对应的文件名列表，如果提供，则会存储文本与文件名的对应关系
        """
        max_characters = truncate_length or int(
            os.getenv("RAG_CHUNK_CHARACTERS", "600")
        )
        overlap_characters = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
        all_chunks: list[str] = []
        all_sources: list[str | None] = []

        for index, text in enumerate(texts):
            source = file_names[index] if file_names and index < len(file_names) else None
            chunks = chunk_text(
                text,
                max_characters=max_characters,
                overlap_characters=min(overlap_characters, max_characters - 1),
            )
            for chunk_index, chunk in enumerate(chunks, start=1):
                all_chunks.append(chunk)
                all_sources.append(
                    f"{source}#chunk{chunk_index}" if source else None
                )

        if not all_chunks:
            return
        if self.model is not None:
            _progress(f"计算 {len(all_chunks)} 个段落感知文本分块的嵌入...")
            embeddings = self.model.encode(all_chunks)
        else:
            _progress(
                f"未加载语义模型，以关键词检索模式保存 {len(all_chunks)} 个文本分块..."
            )
            embeddings = [None] * len(all_chunks)
        for index, chunk in enumerate(tqdm(all_chunks, desc="处理文本分块")):
            self.text_embeddings[chunk] = embeddings[index]
            if all_sources[index]:
                self.file_info[chunk] = all_sources[index]
        self._lexical_index = None
    
    def process_directory(self, directory_path: str, file_pattern: str = "*.txt", 
                          truncate_length: Optional[int] = None, 
                          encoding: str = 'utf-8') -> None:
        """
        处理指定目录下的所有文本文件，将其内容嵌入为embedding
        
        Args:
            directory_path: 目录路径
            file_pattern: 文件匹配模式，默认为"*.txt"
            truncate_length: 截断长度，如果为None则不截断
            encoding: 文件编码，默认为'utf-8'
        """
        # 获取所有匹配的文件
        file_paths = glob.glob(os.path.join(directory_path, file_pattern))
        
        if not file_paths:
            _progress(f"在目录 {directory_path} 中未找到匹配 {file_pattern} 的文件")
            return
        
        _progress(f"找到 {len(file_paths)} 个文件，开始处理...")
        
        # 批量读取所有文件内容
        texts = []
        file_names = []
        
        for file_path in tqdm(file_paths, desc="读取文件"):
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                # 获取文件名（不包含路径）
                file_name = os.path.basename(file_path)
                
                texts.append(content)
                file_names.append(file_name)
            except Exception as e:
                _progress(f"处理文件 {file_path} 时出错: {str(e)}")
        
        # 批量处理所有文本
        if texts:
            _progress(f"开始处理 {len(texts)} 个文本...")
            self.add_texts(texts, truncate_length=truncate_length, file_names=file_names)
            _progress(f"文本处理完成，共生成 {len(self.text_embeddings)} 个嵌入向量")
    
    def save_with_file_info(self, output_dir: str) -> None:
        """
        将文本embedding和文件信息保存到指定目录
        
        Args:
            output_dir: 输出目录路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        _progress(f"保存嵌入向量到 {output_dir}...")
        
        # 保存embedding
        embeddings_path = os.path.join(output_dir, "embeddings.pkl")
        with open(embeddings_path, 'wb') as f:
            pickle.dump(self.text_embeddings, f)
        
        # 保存文件信息
        file_info_path = os.path.join(output_dir, "file_info.pkl")
        with open(file_info_path, 'wb') as f:
            pickle.dump(self.file_info, f)
        
        # 保存文本内容到文本文件，方便查看
        texts_path = os.path.join(output_dir, "texts.txt")
        with open(texts_path, 'w', encoding='utf-8') as f:
            for i, (text, file_name) in enumerate(tqdm(self.file_info.items(), desc="保存文本信息")):
                f.write(f"文件: {file_name}\n")
                f.write(f"内容: {text[:100]}...\n" if len(text) > 100 else f"内容: {text}\n")
                f.write("-" * 80 + "\n")
        
        _progress(f"保存完成，共保存了 {len(self.text_embeddings)} 个嵌入向量")
    
    @classmethod
    def load_with_file_info(cls, input_dir: str, model_name: str = MODELNAME,
                           query_instruction: str = QUERYQUESTION,
                           use_fp16: bool = True) -> 'TextEmbedding':
        """
        从指定目录加载文本embedding和文件信息
        
        Args:
            input_dir: 输入目录路径
            model_name: 使用的模型名称
            query_instruction: 检索指令
            use_fp16: 是否使用fp16精度
            
        Returns:
            TextEmbedding实例
        """
        _progress(f"从 {input_dir} 加载嵌入向量...")
        instance = cls(model_name, query_instruction, use_fp16)
        
        # 加载embedding
        embeddings_path = os.path.join(input_dir, "embeddings.pkl")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                instance.text_embeddings = pickle.load(f)
            _progress(f"已加载 {len(instance.text_embeddings)} 个嵌入向量")
        else:
            _progress(f"未找到嵌入向量文件: {embeddings_path}")
        
        # 加载文件信息
        file_info_path = os.path.join(input_dir, "file_info.pkl")
        if os.path.exists(file_info_path):
            with open(file_info_path, 'rb') as f:
                instance.file_info = pickle.load(f)
            _progress(f"已加载 {len(instance.file_info)} 个文件信息")
        else:
            _progress(f"未找到文件信息文件: {file_info_path}")
        
        instance._check_loaded_vectors()
        return instance
    
    def search_similar_texts(self, query: str, top_n: int = 5, min_similarity: float = 0.5) -> List[Tuple[str, float, Optional[str]]]:
        """
        搜索与查询文本最相似的文本
        
        Args:
            query: 查询文本
            top_n: 返回的最相似文本数量
            min_similarity: 最小相似度阈值
            
        Returns:
            包含(文本, 相似度, 文件名)元组的列表，按相似度降序排列
        """
        if not self.text_embeddings:
            return []
        
        if self.model is None:
            lexical_threshold = float(
                os.getenv("RAG_LEXICAL_MIN_SCORE", "0.05")
            )
            if self._lexical_index is None:
                self._lexical_index = LexicalIndex([
                    (text, self.file_info.get(text)) for text in self.text_embeddings
                ])
            return self._lexical_index.search(
                query,
                top_n=top_n,
                min_score=lexical_threshold,
            )

        _progress(f"计算查询文本 '{query}' 的嵌入向量...")
        # 计算查询文本的embedding
        query_embedding = self.model.encode([query])[0]
        
        # 计算相似度
        _progress("计算相似度...")
        similarities = []
        for text, embedding in tqdm(self.text_embeddings.items(), desc="计算相似度"):
            if embedding is None:
                continue
            similarity = np.dot(query_embedding, embedding)
            if similarity >= min_similarity:
                file_name = self.file_info.get(text)
                similarities.append((text, float(similarity), file_name))
        
        # 扩大候选集后结合关键词覆盖率做二阶段重排序。
        similarities.sort(key=lambda x: x[1], reverse=True)
        similarities = hybrid_rerank(
            query,
            similarities[: max(top_n * 4, top_n)],
            top_n=top_n,
            dense_weight=float(os.getenv("RAG_DENSE_WEIGHT", "0.8")),
        )
        
        _progress(f"找到 {len(similarities)} 个相似度大于 {min_similarity} 的文本")
        
        # 返回前top_n个结果
        return similarities
    
    def save(self, file_path: str) -> None:
        """
        保存文本与embedding对到本地
        
        Args:
            file_path: 保存的文件路径
        """
        _progress(f"保存嵌入向量到 {file_path}...")
        data = {
            'text_embeddings': self.text_embeddings,
            'file_info': self.file_info
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        _progress(f"保存完成，共保存了 {len(self.text_embeddings)} 个嵌入向量")
    
    @classmethod
    def load(cls, file_path: str, model_name: str = MODELNAME,
             query_instruction: str = QUERYQUESTION,
             use_fp16: bool = True) -> 'TextEmbedding':
        """
        从本地加载文本与embedding对
        
        Args:
            file_path: 加载的文件路径
            model_name: 使用的模型名称
            query_instruction: 检索指令
            use_fp16: 是否使用fp16精度
            
        Returns:
            TextEmbedding实例
        """
        _progress(f"从 {file_path} 加载嵌入向量...")
        instance = cls(model_name, query_instruction, use_fp16)
        
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                # 兼容旧版本的保存格式
                if isinstance(data, dict) and 'text_embeddings' in data:
                    instance.text_embeddings = data['text_embeddings']
                    instance.file_info = data.get('file_info', {})
                else:
                    instance.text_embeddings = data
                    instance.file_info = {}
            _progress(f"加载完成，共加载了 {len(instance.text_embeddings)} 个嵌入向量")
        else:
            _progress(f"文件 {file_path} 不存在")
        
        instance._check_loaded_vectors()
        return instance
    
    def get_all_texts(self) -> List[str]:
        """
        获取所有存储的文本
        
        Returns:
            文本列表
        """
        return list(self.text_embeddings.keys())
    
    def get_file_info(self, text: str) -> Optional[str]:
        """
        获取指定文本的文件信息
        
        Args:
            text: 文本
            
        Returns:
            文本对应的文件信息，如果不存在则返回None
        """
        return self.file_info.get(text)
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        获取指定文本的embedding
        
        Args:
            text: 文本
            
        Returns:
            文本对应的embedding，如果不存在则返回None
        """
        return self.text_embeddings.get(text)
    
    def clear(self) -> None:
        """
        清空所有存储的文本与embedding对
        """
        self.text_embeddings.clear()
        self.file_info.clear()
        self._lexical_index = None
        _progress("已清空所有文本和嵌入向量")
    
    def __len__(self) -> int:
        """
        返回存储的文本数量
        """
        return len(self.text_embeddings)

    def get_status(self) -> dict[str, Any]:
        """Return a serializable status without exposing configuration secrets."""
        return {
            "available": bool(self.text_embeddings),
            "backend": self.backend,
            "document_count": len(self.text_embeddings),
            "degraded": self.backend != "semantic",
            "reason": self.degraded_reason,
        }

    def _check_loaded_vectors(self) -> None:
        """Do not advertise semantic search over a lexical-only saved index."""
        if self.model is not None and self.text_embeddings and all(
            vector is None for vector in self.text_embeddings.values()
        ):
            if os.getenv("RAG_BACKEND", "auto").strip().lower() == "semantic":
                raise RuntimeError("知识库没有语义向量，请使用语义后端重新构建索引")
            self.model = None
            self.backend = "lexical"
            self.degraded_reason = "知识库仅包含文本，已降级为关键词检索；语义检索需要重建向量"
