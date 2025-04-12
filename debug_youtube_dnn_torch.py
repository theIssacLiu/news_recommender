import pandas as pd
from tqdm import tqdm
import os
import torch
import numpy as np
import faiss

from data_processing import get_all_click_sample, get_item_info_df
from utils import get_user_item_time, get_item_topk_click, get_item_info_dict
from models.YouTubeDNN_torch import YouTubeDNNModel

data_path = './data_raw/'
save_path = 'cache/'

def debug_youtube_dnn_simple():
    """
    简化版YouTubeDNN调试函数，仅测试模型架构是否正常工作
    """
    print("🛠️ Debug 模式启动：测试PyTorch版YouTubeDNN模型架构")
    
    # 创建一个简单的模型
    user_count = 100
    item_count = 200
    embedding_dim = 16
    
    # 初始化模型
    model = YouTubeDNNModel(user_count, item_count, embedding_dim=embedding_dim)
    print("✅ 模型初始化成功")
    
    # 创建一些随机输入
    batch_size = 4
    max_seq_len = 30
    
    # 随机生成用户ID、历史序列、目标物品和序列长度
    user_ids = torch.randint(0, user_count, (batch_size,))
    hist_item_seq = torch.randint(0, item_count, (batch_size, max_seq_len))
    target_items = torch.randint(0, item_count, (batch_size,))
    seq_lens = torch.randint(1, max_seq_len + 1, (batch_size,))
    
    print(f"✅ 随机输入生成: batch_size={batch_size}, max_seq_len={max_seq_len}")
    print(f"  user_ids shape: {user_ids.shape}")
    print(f"  hist_item_seq shape: {hist_item_seq.shape}")
    print(f"  target_items shape: {target_items.shape}")
    print(f"  seq_lens shape: {seq_lens.shape}")
    
    # 前向传播
    try:
        scores = model(user_ids, hist_item_seq, target_items, seq_lens)
        print(f"✅ 前向传播成功! scores shape: {scores.shape}")
        
        # 获取用户嵌入和物品嵌入
        user_embs = model.get_user_embedding(user_ids, hist_item_seq, seq_lens)
        item_embs = model.get_item_embedding(target_items)
        
        print(f"✅ 用户嵌入提取成功! user_embs shape: {user_embs.shape}")
        print(f"✅ 物品嵌入提取成功! item_embs shape: {item_embs.shape}")
        
        # 测试计算相似度
        sim_scores = torch.matmul(user_embs, item_embs.t())
        print(f"✅ 相似度计算成功! sim_scores shape: {sim_scores.shape}")
        
        print("\n🎉 模型架构测试完成 ✅")
    except Exception as e:
        print(f"❌ 模型测试失败: {str(e)}")
        import traceback
        traceback.print_exc()


def debug_youtube_dnn_recommendation():
    """
    使用PyTorch版的YouTubeDNN模型进行推荐测试
    """
    print("🛠️ Debug 模式启动：测试PyTorch版YouTubeDNN推荐流程")
    
    # 模拟一个小规模的推荐系统
    user_count = 20
    item_count = 50
    embedding_dim = 16
    
    # 初始化模型
    model = YouTubeDNNModel(user_count, item_count, embedding_dim=embedding_dim)
    print("✅ 模型初始化成功")
    
    # 模拟训练过程，此处只是演示，实际应使用真实数据训练
    # 在这个例子中，我们假设模型已经训练好了
    
    # 1. 准备所有物品的嵌入
    all_items = torch.arange(item_count)
    all_item_embs = model.get_item_embedding(all_items).detach().numpy()
    print(f"✅ 计算所有物品嵌入: 共 {item_count} 个物品")
    
    # 2. 使用Faiss构建索引
    index = faiss.IndexFlatIP(embedding_dim)
    # 添加归一化的物品嵌入
    normalized_item_embs = all_item_embs / np.linalg.norm(all_item_embs, axis=1, keepdims=True)
    index.add(np.ascontiguousarray(normalized_item_embs))
    print("✅ Faiss索引构建成功")
    
    # 3. 为用户生成推荐
    # 模拟3个用户的历史交互
    test_users = [0, 5, 10]  # 用户ID
    topk = 5  # 为每个用户推荐5个物品
    
    for user_id in test_users:
        print(f"\n👤 为用户 {user_id} 生成推荐:")
        
        # 模拟用户的历史交互
        # 在实际应用中，应该从数据库或日志中获取用户真实的历史交互
        hist_len = np.random.randint(3, 10)  # 随机生成3-9个历史交互
        hist_items = torch.randint(0, item_count, (1, max(30, hist_len)))
        seq_len = torch.tensor([hist_len])
        user_tensor = torch.tensor([user_id])
        
        # 获取用户嵌入
        user_emb = model.get_user_embedding(user_tensor, hist_items, seq_len).detach().numpy()
        # 归一化用户嵌入
        user_emb = user_emb / np.linalg.norm(user_emb, axis=1, keepdims=True)
        
        # 使用Faiss查找最相似的物品
        sim, idx = index.search(np.ascontiguousarray(user_emb), topk)
        
        # 打印推荐结果
        print(f"  历史交互物品数: {hist_len}")
        for i in range(topk):
            item_id = idx[0][i]
            score = sim[0][i]
            print(f"  📄 推荐物品 {item_id}, 相似度得分: {score:.4f}")
    
    print("\n🎉 推荐测试完成 ✅")


if __name__ == '__main__':
    #debug_youtube_dnn_simple()
    debug_youtube_dnn_recommendation() 