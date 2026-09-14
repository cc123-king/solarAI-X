from embed import TextEmbedding
import os
import shutil
import sys

def check_tqdm():
    """检查是否安装了tqdm库，如果没有则提示安装"""
    try:
        import tqdm
        print("已检测到tqdm库")
    except ImportError:
        print("未检测到tqdm库，建议安装以获得更好的进度显示体验")
        print("可以使用以下命令安装: pip install tqdm")
        print("继续使用内置的简易进度显示...\n")

def main():
    # 检查tqdm库
    check_tqdm()
    
    # 设置目录路径
    example_dir = "/home/gqh/code/project/solarllm/txt"
    output_dir = "/home/gqh/code/project/solarllm/embedding"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化TextEmbedding类
    print("初始化文本嵌入模型...")
    text_embedding = TextEmbedding(
        model_name='BAAI/bge-base-en-v1.5',
        query_instruction="Represent this sentence for searching relevant passages:",
        use_fp16=True
    )
    
    # 处理目录
    print(f"\n开始处理目录 {example_dir} 中的文本文件...")
    text_embedding.process_directory(
        directory_path=example_dir,
        file_pattern="*.txt",
        truncate_length=3000,  # 设置截断长度为3000
        encoding='utf-8'
    )
    
    # 显示处理结果
    print(f"\n处理完成，共处理了 {len(text_embedding)} 个文本片段")
    
    # 保存到输出目录
    print(f"\n开始保存文本嵌入到目录 {output_dir}...")
    text_embedding.save_with_file_info(output_dir)
    
    print(f"\n测试从目录 {output_dir} 加载文本嵌入...")
    loaded_embedding = TextEmbedding.load_with_file_info(output_dir)
    
    # 搜索相似文本
    query = "介绍一下TOPCon电池"
    top_n = 5
    min_similarity = 0
    
    print(f"\n搜索与 '{query}' 相似的文本 (top {top_n}, 最小相似度 {min_similarity}):")
    results = loaded_embedding.search_similar_texts(query, top_n, min_similarity)
    
    if results:
        print("\n搜索结果:")
        for i, (text, similarity, file_name) in enumerate(results):
            print(f"{i+1}. 相似度: {similarity:.4f}, 文件: {file_name}")
            print(f"   文本: {text}")
            print()
    else:
        print("未找到相似文本。")
    
    print("程序执行完毕")

if __name__ == "__main__":
    main() 