import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from utils import get_item_user_time_dict


# 定义用户活跃度权重
def get_user_activate_degree_dict(all_click_df):
    all_click_df_ = all_click_df.groupby('user_id')['click_article_id'].count().reset_index()

    # 用户活跃度归一化
    mm = MinMaxScaler()
    all_click_df_['click_article_id'] = mm.fit_transform(all_click_df_[['click_article_id']])
    user_activate_degree_dict = dict(zip(all_click_df_['user_id'], all_click_df_['click_article_id']))

    return user_activate_degree_dict

# UserCF算法
def usercf_sim(all_click_df, user_activate_degree_dict, save_path, use_cache=True):
    """
    用户相似性矩阵计算 + 缓存机制（支持版本路径）

    :param all_click_df: 用户点击日志
    :param user_activate_degree_dict: 用户活跃度字典
    :param save_path: 缓存路径（可为目录或具体文件名）
    :param use_cache: 是否使用缓存
    :return: 用户相似度矩阵 u2u_sim_
    """

    # === 路径处理：若为目录，拼接默认文件名 ===
    if os.path.splitext(save_path)[1] == '':
        save_path = os.path.join(save_path, 'usercf_u2u_sim.pkl')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === 缓存加载逻辑 ===
    if use_cache and os.path.exists(save_path):
        print(f"[usercf_sim] ✅ 使用缓存文件：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print("[usercf_sim] 🚧 正在重新计算用户相似度矩阵...")

    # === 正式计算 ===
    item_user_time_dict = get_item_user_time_dict(all_click_df)
    u2u_sim = {}
    user_cnt = defaultdict(int)

    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, click_time in user_time_list:
            user_cnt[u] += 1
            u2u_sim.setdefault(u, {})
            for v, click_time in user_time_list:
                if u == v:
                    continue
                u2u_sim[u].setdefault(v, 0)
                # 用户活跃度加权（可调）
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

    # === 归一化 ===
    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # === 保存相似度矩阵 ===
    with open(save_path, 'wb') as f:
        pickle.dump(u2u_sim_, f)

    print(f"[usercf_sim] ✅ 相似度矩阵已保存至：{save_path}")
    return u2u_sim_



# 基于用户的召回 u2u2i
def user_based_recommend(user_id, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                         item_topk_click, item_created_time_dict, emb_i2i_sim):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param u2u_sim: 字典，文章相似性矩阵
        :param sim_user_topk: 整数， 选择与当前用户最相似的前k个用户
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        :param item_created_time_dict: 文章创建时间列表
        :param emb_i2i_sim: 字典基于内容embedding算的文章相似矩阵

        return: 召回的文章列表 {item1:score1, item2: score2...}
    """
    # 历史交互
    user_item_time_list = user_item_time_dict[user_id]  # {item1: time1, item2: time2...}
    user_hist_items = set([i for i, t in user_item_time_list])  # 存在一个用户与某篇文章的多次交互， 这里得去重

    items_rank = {}
    for sim_u, wuv in sorted(u2u_sim[user_id].items(), key=lambda x: x[1], reverse=True)[:sim_user_topk]:
        for i, click_time in user_item_time_dict[sim_u]:
            if i in user_hist_items:
                continue
            items_rank.setdefault(i, 0)

            loc_weight = 1.0
            content_weight = 1.0
            created_time_weight = 1.0

            # 当前文章与该用户看的历史文章进行一个权重交互
            for loc, (j, click_time) in enumerate(user_item_time_list):
                # 点击时的相对位置权重
                loc_weight += 0.9 ** (len(user_item_time_list) - loc)
                # 内容相似性权重
                if emb_i2i_sim.get(i, {}).get(j, None) is not None:
                    content_weight += emb_i2i_sim[i][j]
                if emb_i2i_sim.get(j, {}).get(i, None) is not None:
                    content_weight += emb_i2i_sim[j][i]

                # 创建时间差权重
                created_time_weight += np.exp(0.8 * np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

            items_rank[i] += loc_weight * content_weight * created_time_weight * wuv

    # 热度补全
    if len(items_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in items_rank.items():  # 填充的item应该不在原来的列表中
                continue
            items_rank[item] = - i - 100  # 随便给个复数就行
            if len(items_rank) == recall_item_num:
                break

    items_rank = sorted(items_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return items_rank

# ============================
# ✅ UserCF相似度计算（传统相似用户）
# ============================
def generate_usercf_recall_dict(click_df, user_item_time_dict, u2u_sim, sim_user_topk, recall_item_num,
                  item_topk_click, item_created_time_dict, emb_i2i_sim):
    user_recall_items_dict = {}
    for user in tqdm(click_df['user_id'].unique()):
        rec_items = user_based_recommend(user, user_item_time_dict, u2u_sim, sim_user_topk,
                                         recall_item_num, item_topk_click, item_created_time_dict, emb_i2i_sim)
        user_recall_items_dict[user] = rec_items
    return user_recall_items_dict


# ============================
# ✅ User Embedding 相似度计算（YouTubeDNN用户向量）
# ============================
def generate_ucercf_embedding_recall_dict(click_df, user_emb_dict, save_path, topk):
    user_list = []
    user_emb_list = []
    for user_id, user_emb in user_emb_dict.items():
        user_list.append(user_id)
        user_emb_list.append(user_emb)

    user_index_2_rawid_dict = {k: v for k, v in zip(range(len(user_list)), user_list)}
    user_emb_np = np.array(user_emb_list, dtype=np.float32)

    index = faiss.IndexFlatIP(user_emb_np.shape[1])
    index.add(user_emb_np)
    sim, idx = index.search(user_emb_np, topk)

    user_sim_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(range(len(user_emb_np)), sim, idx)):
        target_raw_id = user_index_2_rawid_dict[target_idx]
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = user_index_2_rawid_dict[rele_idx]
            user_sim_dict[target_raw_id][rele_raw_id] = user_sim_dict.get(target_raw_id, {}).get(rele_raw_id, 0) + sim_value

    with open(os.path.join(save_path, 'youtube_u2u_sim.pkl'), 'wb') as f:
        pickle.dump(user_sim_dict, f)

    return user_sim_dict
