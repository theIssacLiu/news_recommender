import os
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import faiss
import torch
from datetime import datetime

from utils import (
    get_user_item_time,
    get_item_topk_click,
    get_item_info_dict
)
from data_processing import get_all_click_df, get_item_info_df, embdding_sim
from models.itemcf_recall import itemcf_sim, generate_itemcf_recall_dict, generate_itemcf_embedding_recall_dict
from models.usercf_recall import usercf_sim, generate_usercf_recall_dict, get_user_activate_degree_dict, u2u_embedding_sim
from models.YouTubeDNN_torch import YouTubeDNNModel, youtubednn_u2i_dict
from submission import submit

# 直接使用相对路径
data_path = './data_raw/'
cache_dir = './cache/online'  # 改为相对路径

# 确保缓存目录存在
os.makedirs(cache_dir, exist_ok=True)

print(f"缓存目录: {cache_dir}")

def get_youtube_recall(train_df, test_df, save_path, use_cache=False, epochs=10, batch_size=64, embedding_dim=64, recall_num=50):
    """
    使用PyTorch版本的YouTubeDNN模型生成用户-物品召回表
    
    Args:
        train_df: 训练数据
        test_df: 测试数据
        save_path: 结果保存路径
        use_cache: 是否使用缓存
        epochs: 训练轮数
        batch_size: 批大小
        embedding_dim: 嵌入维度
        recall_num: 召回数量
        
    Returns:
        用户-物品召回表，格式为{用户ID: [(物品ID, 得分), ...]}
    """
    # 定义相关缓存路径
    cache_path = os.path.join(save_path, 'youtube_u2i_dict.pkl')
    model_path = os.path.join(save_path, 'youtube_dnn_model.pth')
    user_emb_path = os.path.join(save_path, 'user_youtube_emb.pkl') 
    item_emb_path = os.path.join(save_path, 'item_youtube_emb.pkl')
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 检查召回结果缓存
    if use_cache and os.path.exists(cache_path):
        print(f"[get_youtube_recall] ✅ 使用缓存：{cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("[get_youtube_recall] 🚀 生成YouTubeDNN召回结果...")
    
    # 仅使用用户和物品ID，简化处理
    df = pd.concat([train_df, test_df], ignore_index=True)
    
    # 重映射用户和物品ID到连续空间，避免索引问题
    user_id_map = {uid: idx for idx, uid in enumerate(df['user_id'].unique())}
    item_id_map = {iid: idx for idx, iid in enumerate(df['click_article_id'].unique())}
    
    # 记录逆映射，用于后续还原ID
    user_id_reverse_map = {idx: uid for uid, idx in user_id_map.items()}
    item_id_reverse_map = {idx: iid for iid, idx in item_id_map.items()}
    
    # 映射后的ID范围
    user_count = len(user_id_map)
    item_count = len(item_id_map)
    
    print(f"[get_youtube_recall] 用户数量: {user_count}, 物品数量: {item_count}")
    
    # 获取用户历史交互，使用映射后的ID
    user_hist_dict = {}
    for user_id, group in df.groupby('user_id'):
        mapped_user_id = user_id_map[user_id]
        mapped_items = [item_id_map[item] for item in group.sort_values('click_timestamp')['click_article_id'].tolist()]
        user_hist_dict[mapped_user_id] = mapped_items
    
    # 创建模型
    model = YouTubeDNNModel(user_count, item_count, embedding_dim=embedding_dim)
    
    # 检查模型缓存
    if use_cache and os.path.exists(model_path):
        print(f"[get_youtube_recall] ✅ 加载预训练模型：{model_path}")
        model.load_state_dict(torch.load(model_path))
    
    # 检查嵌入缓存
    if use_cache and os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
        print(f"[get_youtube_recall] ✅ 加载用户和物品嵌入")
        with open(user_emb_path, 'rb') as f:
            user_embeddings = pickle.load(f)
        with open(item_emb_path, 'rb') as f:
            item_embeddings = pickle.load(f)
    else:
        # 生成用户和物品的嵌入
        print("[get_youtube_recall] 计算用户和物品嵌入...")
        model.eval()
        
        # 为所有物品生成嵌入（使用映射后的ID）
        with torch.no_grad():
            all_item_ids = torch.LongTensor(list(range(item_count)))
            all_item_embs = model.get_item_embedding(all_item_ids).detach().numpy()
            normalized_item_embs = all_item_embs / np.linalg.norm(all_item_embs, axis=1, keepdims=True)
            
            # 保存物品嵌入字典，使用原始ID
            item_embeddings = {item_id_reverse_map[idx]: emb for idx, emb in enumerate(normalized_item_embs)}
            with open(item_emb_path, 'wb') as f:
                pickle.dump(item_embeddings, f)
        
        # 计算用户嵌入
        user_embeddings = {}
        max_seq_len = 30
        
        with torch.no_grad():
            for mapped_user_id, hist_items in tqdm(user_hist_dict.items(), desc="计算用户嵌入"):
                if not hist_items:
                    continue
                    
                # 最多使用最近30个交互
                hist_items = hist_items[-max_seq_len:] if len(hist_items) > max_seq_len else hist_items
                hist_len = len(hist_items)
                
                # 将历史交互转换为模型输入
                hist_tensor = torch.LongTensor(hist_items + [0] * (max_seq_len - hist_len))
                hist_tensor = hist_tensor.unsqueeze(0)  # 增加批次维度
                user_tensor = torch.LongTensor([mapped_user_id])
                seq_len = torch.LongTensor([hist_len])
                
                # 获取用户嵌入
                try:
                    user_emb = model.get_user_embedding(user_tensor, hist_tensor, seq_len).numpy()
                    # 保存时使用原始ID
                    original_user_id = user_id_reverse_map[mapped_user_id]
                    user_embeddings[original_user_id] = user_emb.squeeze() / np.linalg.norm(user_emb)
                except Exception as e:
                    print(f"[get_youtube_recall] ⚠️ 处理用户 {mapped_user_id} 嵌入时出错: {str(e)}")
                    continue
        
        # 保存用户嵌入
        with open(user_emb_path, 'wb') as f:
            pickle.dump(user_embeddings, f)
    
    # 准备向量检索
    print("[get_youtube_recall] 使用Faiss进行向量检索...")
    user_ids = list(user_embeddings.keys())
    user_embs = np.array([user_embeddings[user_id] for user_id in user_ids], dtype=np.float32)
    
    item_ids = list(item_embeddings.keys())
    item_embs = np.array([item_embeddings[item_id] for item_id in item_ids], dtype=np.float32)
    
    # 构建索引
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(np.ascontiguousarray(item_embs))
    
    # 为测试集中的用户生成召回结果
    user_recall_items_dict = {}
    topk = recall_num  # 每个用户召回指定数量的文章
    
    # 执行向量检索
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)
    
    # 构建用户召回结果
    for i, user_id in enumerate(user_ids):
        item_list = []
        for j, item_idx in enumerate(idx[i]):
            if item_idx < len(item_ids):
                item_id = item_ids[item_idx]
                score = float(sim[i][j])
                item_list.append((item_id, score))
        user_recall_items_dict[user_id] = item_list
    
    # 保存召回结果
    with open(cache_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)
    
    print(f"[get_youtube_recall] ✅ 召回结果已保存至：{cache_path}")
    return user_recall_items_dict

def combine_recall_results(user_multi_recall_dict, weight_dict=None, topk=25, save_path='cache/'):
    """
    合并多路召回结果
    
    Args:
        user_multi_recall_dict: 多路召回结果字典，格式为 {召回方法: {用户ID: [(物品ID, 得分), ...]}}
        weight_dict: 各路召回的权重字典，格式为 {召回方法: 权重值}
        topk: 最终返回的推荐物品数量
        save_path: 结果保存路径
        
    Returns:
        合并后的召回结果字典，格式为 {用户ID: [(物品ID, 得分), ...]}
    """
    final_recall_items_dict = {}
    
    # 对每一种召回结果按照用户进行归一化，方便后面多种召回结果，相同用户的物品之间权重相加
    def norm_user_recall_items_sim(sorted_item_list):
        # 如果冷启动中没有文章或者只有一篇文章，直接返回，出现这种情况的原因可能是冷启动召回的文章数量太少了，
        # 基于规则筛选之后就没有文章了, 这里还可以做一些其他的策略性的筛选
        if len(sorted_item_list) < 2:
            return sorted_item_list
        
        min_sim = sorted_item_list[-1][1]
        max_sim = sorted_item_list[0][1]
        
        norm_sorted_item_list = []
        for item, score in sorted_item_list:
            if max_sim > 0:
                norm_score = 1.0 * (score - min_sim) / (max_sim - min_sim) if max_sim > min_sim else 1.0
            else:
                norm_score = 0.0
            norm_sorted_item_list.append((item, norm_score))
            
        return norm_sorted_item_list
    
    print('多路召回合并...')
    for method, user_recall_items in tqdm(user_multi_recall_dict.items()):
        print(method + '...')
        # 在计算最终召回结果的时候，也可以为每一种召回结果设置一个权重
        if weight_dict is None:
            recall_method_weight = 1
        else:
            recall_method_weight = weight_dict.get(method, 1)
        
        for user_id, sorted_item_list in user_recall_items.items(): # 进行归一化
            user_recall_items[user_id] = norm_user_recall_items_sim(sorted_item_list)
        
        for user_id, sorted_item_list in user_recall_items.items():
            final_recall_items_dict.setdefault(user_id, {})
            for item, score in sorted_item_list:
                final_recall_items_dict[user_id].setdefault(item, 0)
                final_recall_items_dict[user_id][item] += recall_method_weight * score  
    
    final_recall_items_dict_rank = {}
    # 多路召回时也可以控制最终的召回数量
    for user, recall_item_dict in final_recall_items_dict.items():
        final_recall_items_dict_rank[user] = sorted(recall_item_dict.items(), key=lambda x: x[1], reverse=True)[:topk]

    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 将多路召回后的最终结果字典保存到本地
    with open(save_path, 'wb') as f:
        pickle.dump(final_recall_items_dict_rank, f)

    return final_recall_items_dict_rank

def online_predict(use_cache=True, recall_num=50, epochs=10, batch_size=32, embedding_dim=32):
    print(f"\n📂 当前使用的缓存目录: {cache_dir}")
    
    # 检查现有缓存文件
    if os.path.exists(cache_dir):
        cache_files = os.listdir(cache_dir)
        print("\n现有缓存文件:")
        for file in cache_files:
            file_path = os.path.join(cache_dir, file)
            file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  - {file} (更新时间: {file_time})")
    
    # 统一使用一个模型文件名
    model_cache = os.path.join(cache_dir, 'youtube_model.pth')  # 使用实际保存的文件名
    
    # 如果模型缓存存在，直接加载
    if use_cache and os.path.exists(model_cache):
        print(f"\n✅ 发现模型缓存: {model_cache}")
        print(f"   缓存时间: {datetime.fromtimestamp(os.path.getmtime(model_cache))}")
        print("✅ 已成功找到模型缓存")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Step 1: 加载数据
    print("📌 加载训练数据和测试数据...")
    train_df = get_all_click_df(data_path=data_path, offline=True)  # 训练数据
    test_df = pd.read_csv(data_path + 'testA_click_log.csv')  # 测试数据
    print(f"✅ 训练集 {len(train_df)} 条，测试集 {len(test_df)} 条")
    
    # Step 2: 准备相关数据
    item_info_df = get_item_info_df(data_path)
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)
    user_item_time_dict = get_user_item_time(train_df)
    item_topk_click = get_item_topk_click(train_df, k=50)
    
    # Step 3: 加载并抽样 embedding，构建 embedding 相似度
    print("🚀 Step 3：加载文章 embedding 并抽样构建相似度")
    emb_sample_n = 1000
    item_emb_df = pd.read_csv(data_path + '/articles_emb.csv').sample(n=emb_sample_n, random_state=42)
    emb_item_ids = set(item_emb_df['article_id'])
    click_df_for_emb = train_df[train_df['click_article_id'].isin(emb_item_ids)]
    
    # embedding_sim 只用 click_df_for_emb，而不要污染主流程的 train_df
    emb_i2i_sim = embdding_sim(click_df_for_emb, item_emb_df, save_path=cache_dir, topk=10)
    print(f"✅ Step 3：完成 embedding 相似度计算")
    
    # Step 4: 首先生成YouTubeDNN召回并提取用户嵌入
    print("🔄 Step 4.1: 生成YouTubeDNN召回并提取用户嵌入...")
    
    # 合并数据并增加时间戳排序
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df = all_df.sort_values('click_timestamp')
    
    # 先生成YouTubeDNN的召回结果和嵌入
    youtube_recall_dict = youtubednn_u2i_dict(
        data=all_df,
        save_path=cache_dir,
        topk=recall_num,
        epochs=epochs,
        batch_size=batch_size,
        embedding_dim=embedding_dim
    )
    
    # 然后加载生成的用户嵌入
    user_emb_path = os.path.join(cache_dir, 'youtube_embeddings.pkl')  # 注意这里改用正确的文件名
    if os.path.exists(user_emb_path):
        print(f"[online_predict] ✅ 加载用户嵌入：{user_emb_path}")
        with open(user_emb_path, 'rb') as f:
            cache_data = pickle.load(f)
            user_embeddings = cache_data['user_embeddings']
    else:
        print("[online_predict] ⚠️ 无法找到用户嵌入文件")
        user_embeddings = {}
    
    # Step 5: 生成传统ItemCF召回
    print("🔄 Step 4.2: 生成传统ItemCF召回...")
    
    # 合并训练集和测试集来获取完整的用户-物品交互字典
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_user_item_time_dict = get_user_item_time(all_df)
    
    # 使用合并后的数据计算物品相似度
    i2i_sim = itemcf_sim(
        all_df,  # 使用合并后的数据
        item_created_time_dict,
        save_path=os.path.join(cache_dir, 'itemcf_i2i_sim.pkl'),
        use_cache=use_cache
    )
    
    itemcf_recall_dict = generate_itemcf_recall_dict(
        val_df=test_df,
        user_item_time_dict=all_user_item_time_dict,
        i2i_sim=i2i_sim,
        sim_item_topk=10,
        recall_item_num=recall_num,
        item_topk_click=item_topk_click,
        item_created_time_dict=item_created_time_dict,
        emb_i2i_sim=emb_i2i_sim,
        save_path=os.path.join(cache_dir, 'itemcf_recall_dict.pkl'),
        use_cache=use_cache
    )
    
    # Step 6: 生成基于Embedding的ItemCF召回
    print("🔄 Step 4.3: 生成基于Embedding的ItemCF召回...")
    itemcf_emb_recall_dict = generate_itemcf_embedding_recall_dict(
        val_df=test_df,
        emb_i2i_sim=emb_i2i_sim,  # 直接使用embedding相似度
        user_item_time_dict=all_user_item_time_dict,  # 使用完整的用户-物品交互字典
        sim_item_topk=10,
        recall_item_num=recall_num,
        item_topk_click=item_topk_click,
        item_created_time_dict=item_created_time_dict,
        save_path=os.path.join(cache_dir, 'itemcf_emb_recall_dict.pkl'),
        use_cache=use_cache
    )
    
    # Step 7: 生成基于Embedding的UserCF召回
    print("\n🔄 Step 4.4: 生成基于Embedding的UserCF召回...")
    
    # 检查用户嵌入的可用性
    if not user_embeddings:
        print("⚠️ 警告：用户嵌入为空")
        print(f"用户嵌入数量: {len(user_embeddings)}")
    
    u2u_emb_sim = u2u_embedding_sim(
        click_df=test_df,  # 注意这里使用test_df而不是val_df
        user_emb_dict=user_embeddings,
        save_path=os.path.join(cache_dir, 'youtube_u2u_sim.pkl'),
        topk=20,
        use_cache=use_cache
    )
    
    # 打印用户相似度矩阵的统计信息
    print(f"\n用户相似度矩阵统计:")
    print(f"总用户数: {len(test_df['user_id'].unique())}")
    print(f"相似度矩阵中的用户数: {len(u2u_emb_sim)}")
    
    print("\n🔄 Step 4.5: 生成基于Embedding的UserCF的召回结果...")
    usercf_emb_recall_dict = generate_usercf_recall_dict(
        click_df=test_df,  # 注意这里使用test_df而不是val_df
        user_item_time_dict=user_item_time_dict,
        u2u_sim=u2u_emb_sim,
        sim_user_topk=10,
        recall_item_num=recall_num,
        item_topk_click=item_topk_click,
        item_created_time_dict=item_created_time_dict,
        emb_i2i_sim=i2i_sim,
        save_path=cache_dir,  # 添加缓存路径
        use_cache=use_cache   # 使用缓存参数
    )
    
    # Step 8: 合并召回结果
    print("\n🔄 合并多路召回结果...")
    # 创建多路召回字典
    user_multi_recall_dict = {
        'itemcf': itemcf_recall_dict,
        'itemcf_emb': itemcf_emb_recall_dict,
        'usercf_emb': usercf_emb_recall_dict,
        'youtube': youtube_recall_dict
    }
    
    # 设置各路召回的权重
    weight_dict = {
        'itemcf': 1.0,
        'itemcf_emb': 0.8,
        'usercf_emb': 0.9,
        'youtube': 1.2
    }
    
    # 合并召回结果
    final_recall_dict = combine_recall_results(
        user_multi_recall_dict=user_multi_recall_dict,
        weight_dict=weight_dict,
        topk=recall_num,
        save_path=os.path.join(cache_dir, 'final_recall_dict.pkl')
    )
    
    # Step 9: 生成提交文件
    print("\n📝 生成提交文件...")
    # 将召回结果转换为DataFrame格式
    recall_list = []
    for user_id, items in final_recall_dict.items():
        # 只保留testA中的用户（200000~249999）
        if 200000 <= user_id <= 249999:
            for item_id, score in items:
                recall_list.append({
                    'user_id': user_id,
                    'click_article_id': item_id,
                    'pred_score': score
                })
    
    recall_df = pd.DataFrame(recall_list)
    
    # 确保results目录存在
    os.makedirs('./results', exist_ok=True)
    
    # 使用submission.py中的submit函数生成提交文件
    submit(
        recall_df=recall_df,
        save_path='./results/',  # 修改保存路径为results文件夹
        topk=5,
        model_name='multi_recall'
    )
    
    print("✅ 提交文件生成完成！")

if __name__ == '__main__':
    # 配置参数
    use_cache = False  # 强制重新训练
    recall_num = 50   # 召回数量
    epochs = 20        # 增加训练轮数
    batch_size = 128   # 增加批次大小
    embedding_dim = 64  # 增加嵌入维度
    
    # 检查设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    # 运行在线预测
    online_predict(
        use_cache=use_cache,  # 强制重新训练
        recall_num=recall_num,
        epochs=epochs,        # 增加训练轮数
        batch_size=batch_size,   # 增加批次大小
        embedding_dim=embedding_dim  # 增加嵌入维度
    ) 