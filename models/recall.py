import math
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import pickle
import os

from utils import get_user_item_time

def itemcf_sim(df, item_created_time_dict, save_path, use_cache=True):
    """
        文章与文章之间的相似性矩阵计算
        :param df: 数据表
        :item_created_time_dict:  文章创建时间的字典
        return : 文章与文章的相似性矩阵
        思路: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
        可自动缓存和复用已有结果
    """
    # 如果是目录路径，就拼接默认文件名
    if os.path.splitext(save_path)[1] == '':
        # 没有扩展名，说明是目录
        save_path = os.path.join(save_path, 'itemcf_i2i_sim.pkl')

    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 缓存判断
    if use_cache and os.path.exists(save_path):
        print(f"[itemcf_sim] ✅ 使用缓存文件：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    # 否则重新计算
    print(f"[itemcf_sim] 重新计算相似度矩阵...")

    user_item_time_dict = get_user_item_time(df)

    # 计算物品相似度
    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        # 在基于商品的协同过滤优化的时候可以考虑时间因素
        for loc1, (i, i_click_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for loc2, (j, j_click_time) in enumerate(item_time_list):
                if i == j:
                    continue
                # 正向/反向点击顺序区分（位置关系）
                loc_alpha = 1.0 if loc2 > loc1 else 0.7
                # 位置权重（点击顺序越近，越相关）其中的参数可以调节
                loc_weight = loc_alpha * (0.9 ** (np.abs(loc2 - loc1) - 1))
                # 点击时间相近权重（可理解为 session 内更相关）其中的参数可以调节
                click_time_weight = np.exp(0.7 ** np.abs(i_click_time - j_click_time))
                # 文章发布时间相近权重（防止跨年代推荐）其中的参数可以调节
                created_time_weight = np.exp(0.8 ** np.abs(item_created_time_dict[i] - item_created_time_dict[j]))

                i2i_sim[i].setdefault(j, 0)
                # 考虑多种因素的权重计算最终的文章之间的相似度
                i2i_sim[i][j] += loc_weight * click_time_weight * created_time_weight / math.log(
                    len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # 将得到的相似性矩阵保存到本地
    with open(save_path, 'wb') as f:
        pickle.dump(i2i_sim_, f)
    print(f"[itemcf_sim] 相似度矩阵已保存至：{save_path}")

    return i2i_sim_


# 基于商品的召回i2i
def item_based_recommend(user_id, user_item_time_dict, i2i_sim, sim_item_topk, recall_item_num, item_topk_click):
    """
        基于文章协同过滤的召回
        :param user_id: 用户id
        :param user_item_time_dict: 字典, 根据点击时间获取用户的点击文章序列   {user1: {item1: time1, item2: time2..}...}
        :param i2i_sim: 字典，文章相似性矩阵
        :param sim_item_topk: 整数， 选择与当前文章最相似的前k篇文章
        :param recall_item_num: 整数， 最后的召回文章数量
        :param item_topk_click: 列表，点击次数最多的文章列表，用户召回补全
        return: 召回的文章列表 {item1:score1, item2: score2...}
        注意: 基于物品的协同过滤(详细请参考上一期推荐系统基础的组队学习)， 在多路召回部分会加上关联规则的召回策略
    """

    # 获取用户历史交互的文章
    user_hist_items = user_item_time_dict[user_id]

    item_rank = {}
    clicked_items = set([i for i, _ in user_hist_items])
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in clicked_items:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank:  # ✅ 判断 item 是否已存在
                continue
            item_rank[item] = - i - 100  # 随便给个负数就行
            if len(item_rank) == recall_item_num:
                break

    item_rank = sorted(item_rank.items(), key=lambda x: x[1], reverse=True)[:recall_item_num]

    return item_rank

def generate_user_recall_dict(val_df,
                               user_item_time_dict,
                               i2i_sim,
                               sim_item_topk,
                               recall_item_num,
                               item_topk_click,
                               save_path='cache/user_recall_items_default.pkl',
                               use_cache=True):
    """
    生成用户的召回列表，并可自动缓存和复用已有结果

    :param val_df: 验证集（用于获取 user_id 列表）
    :param user_item_time_dict: 用户-文章点击时间字典
    :param i2i_sim: 相似度矩阵
    :param sim_item_topk: 每个历史文章选出的相似文章个数
    :param recall_item_num: 最终每个用户召回的文章数
    :param item_topk_click: 热门文章列表（用于召回补全）
    :param save_path: 召回结果缓存路径或目录
    :param use_cache: 是否使用已有缓存
    :return: user_recall_items_dict
    """
    # 如果是目录，拼接默认文件名
    if os.path.splitext(save_path)[1] == '':
        save_path = os.path.join(save_path, 'user_recall_items_default.pkl')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if use_cache and os.path.exists(save_path):
        print(f"[generate_user_recall_dict] ✅ 使用缓存：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print("[generate_user_recall_dict] 🚀 正在生成用户召回列表...")

    user_recall_items_dict = {}
    for user in tqdm(val_df['user_id'].unique()):
        rec_items = item_based_recommend(
            user,
            user_item_time_dict,
            i2i_sim,
            sim_item_topk,
            recall_item_num,
            item_topk_click
        )
        user_recall_items_dict[user] = rec_items

    with open(save_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)

    print(f"[generate_user_recall_dict] ✅ 召回列表保存成功：{save_path}")
    return user_recall_items_dict


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
    用户相似性矩阵计算 + 缓存机制
    :param all_click_df: 用户点击日志
    :param user_activate_degree_dict: 用户活跃度字典
    :param save_path: 缓存路径（可包含版本名）
    :param use_cache: 是否使用缓存
    :return: 用户-用户相似度矩阵 u2u_sim_
    """
    # === 处理路径 ===
    if os.path.splitext(save_path)[1] == '':  # 如果是目录则拼接默认文件名
        save_path = os.path.join(save_path, 'usercf_u2u_sim.pkl')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === 缓存机制 ===
    if use_cache and os.path.exists(save_path):
        print(f"[usercf_sim] ✅ 使用缓存文件：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    print("[usercf_sim] 🚧 正在计算用户相似度矩阵...")

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
                activate_weight = 100 * 0.5 * (user_activate_degree_dict[u] + user_activate_degree_dict[v])
                u2u_sim[u][v] += activate_weight / math.log(len(user_time_list) + 1)

    u2u_sim_ = u2u_sim.copy()
    for u, related_users in u2u_sim.items():
        for v, wij in related_users.items():
            u2u_sim_[u][v] = wij / math.sqrt(user_cnt[u] * user_cnt[v])

    # === 保存缓存 ===
    with open(save_path, 'wb') as f:
        pickle.dump(u2u_sim_, f)
    print(f"[usercf_sim] ✅ 相似度矩阵已保存至：{save_path}")

    return u2u_sim_
