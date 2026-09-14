import os
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional, Union, Any
import glob
from tqdm import tqdm

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
        if FlagAutoModel is None:
            raise RuntimeError(
                "缺少FlagEmbedding，知识库生成和检索功能暂不可用。"
            )

        self.model = FlagAutoModel.from_finetuned(model_name,
                                                 query_instruction_for_retrieval=query_instruction,
                                                 use_fp16=use_fp16)
        self.text_embeddings: Dict[str, np.ndarray] = {}
        # 添加文件信息存储
        self.file_info: Dict[str, str] = {}  # 文本 -> 文件名映射
        
    def add_texts(self, texts: List[str], truncate_length: Optional[int] = None, file_names: Optional[List[str]] = None) -> None:
        """
        批量添加文本并计算embedding
        
        Args:
            texts: 文本列表
            truncate_length: 截断长度，如果为None则不截断
            file_names: 对应的文件名列表，如果提供，则会存储文本与文件名的对应关系
        """
        if truncate_length is None:
            # 不截断，直接计算embedding
            print("计算文本嵌入...")
            embeddings = self.model.encode(texts)
            for i, text in enumerate(tqdm(texts, desc="处理文本")):
                self.text_embeddings[text] = embeddings[i]
                # 如果提供了文件名，则存储文本与文件名的对应关系
                if file_names and i < len(file_names):
                    self.file_info[text] = file_names[i]
        else:
            # 需要截断处理
            print("处理文本并截断...")
            for i, text in enumerate(tqdm(texts, desc="处理文本")):
                file_name = file_names[i] if file_names and i < len(file_names) else None
                
                if len(text) <= truncate_length:
                    # 文本长度小于截断长度，直接计算embedding
                    embedding = self.model.encode([text])[0]
                    self.text_embeddings[text] = embedding
                    if file_name:
                        self.file_info[text] = file_name
                else:
                    # 文本长度大于截断长度，需要截断
                    chunks = []
                    for j in range(0, len(text), truncate_length):
                        chunk = text[j:j+truncate_length]
                        chunks.append(chunk)
                    
                    print(f"文本 {i+1}/{len(texts)} 被截断为 {len(chunks)} 个片段")
                    # 计算每个chunk的embedding
                    chunk_embeddings = self.model.encode(chunks)
                    for j, chunk in enumerate(tqdm(chunks, desc=f"处理文本 {i+1} 的片段")):
                        self.text_embeddings[chunk] = chunk_embeddings[j]
                        if file_name:
                            # 为每个chunk添加文件信息和chunk索引
                            self.file_info[chunk] = f"{file_name}#chunk{j+1}"
    
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
            print(f"在目录 {directory_path} 中未找到匹配 {file_pattern} 的文件")
            return
        
        print(f"找到 {len(file_paths)} 个文件，开始处理...")
        
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
                print(f"处理文件 {file_path} 时出错: {str(e)}")
        
        # 批量处理所有文本
        if texts:
            print(f"开始处理 {len(texts)} 个文本...")
            self.add_texts(texts, truncate_length=truncate_length, file_names=file_names)
            print(f"文本处理完成，共生成 {len(self.text_embeddings)} 个嵌入向量")
    
    def save_with_file_info(self, output_dir: str) -> None:
        """
        将文本embedding和文件信息保存到指定目录
        
        Args:
            output_dir: 输出目录路径
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"保存嵌入向量到 {output_dir}...")
        
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
        
        print(f"保存完成，共保存了 {len(self.text_embeddings)} 个嵌入向量")
    
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
        print(f"从 {input_dir} 加载嵌入向量...")
        instance = cls(model_name, query_instruction, use_fp16)
        
        # 加载embedding
        embeddings_path = os.path.join(input_dir, "embeddings.pkl")
        if os.path.exists(embeddings_path):
            with open(embeddings_path, 'rb') as f:
                instance.text_embeddings = pickle.load(f)
            print(f"已加载 {len(instance.text_embeddings)} 个嵌入向量")
        else:
            print(f"未找到嵌入向量文件: {embeddings_path}")
        
        # 加载文件信息
        file_info_path = os.path.join(input_dir, "file_info.pkl")
        if os.path.exists(file_info_path):
            with open(file_info_path, 'rb') as f:
                instance.file_info = pickle.load(f)
            print(f"已加载 {len(instance.file_info)} 个文件信息")
        else:
            print(f"未找到文件信息文件: {file_info_path}")
        
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
        
        print(f"计算查询文本 '{query}' 的嵌入向量...")
        # 计算查询文本的embedding
        query_embedding = self.model.encode([query])[0]
        
        # 计算相似度
        print("计算相似度...")
        similarities = []
        for text, embedding in tqdm(self.text_embeddings.items(), desc="计算相似度"):
            similarity = np.dot(query_embedding, embedding)
            if similarity >= min_similarity:
                file_name = self.file_info.get(text)
                similarities.append((text, float(similarity), file_name))
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"找到 {len(similarities)} 个相似度大于 {min_similarity} 的文本")
        
        # 返回前top_n个结果
        return similarities[:top_n]
    
    def save(self, file_path: str) -> None:
        """
        保存文本与embedding对到本地
        
        Args:
            file_path: 保存的文件路径
        """
        print(f"保存嵌入向量到 {file_path}...")
        data = {
            'text_embeddings': self.text_embeddings,
            'file_info': self.file_info
        }
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"保存完成，共保存了 {len(self.text_embeddings)} 个嵌入向量")
    
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
        print(f"从 {file_path} 加载嵌入向量...")
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
            print(f"加载完成，共加载了 {len(instance.text_embeddings)} 个嵌入向量")
        else:
            print(f"文件 {file_path} 不存在")
        
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
        print("已清空所有文本和嵌入向量")
    
    def __len__(self) -> int:
        """
        返回存储的文本数量
        """
        return len(self.text_embeddings)
