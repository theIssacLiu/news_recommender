import math
from tqdm import tqdm
from collections import defaultdict
import pickle
import os

from utils import get_user_item_time
import os

def itemcf_sim(df, item_created_time_dict=None, save_path='cache/itemcf_sim.pkl', use_cache=True):
    """
    基于商品协同过滤（无加权项）版本 + 缓存机制

    :param df: 点击数据 DataFrame
    :param save_path: 缓存路径（可以是目录或具体路径）
    :param use_cache: 是否使用缓存
    :return: i2i_sim 字典
    """


    # === 处理保存路径 ===
    if os.path.splitext(save_path)[1] == '':
        save_path = os.path.join(save_path, 'itemcf_i2i_sim_baseline.pkl')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === 使用缓存 ===
    if use_cache and os.path.exists(save_path):
        print(f"[itemcf_sim_baseline] ✅ 使用缓存文件：{save_path}")
        with open(save_path, 'rb') as f:
            return pickle.load(f)

    # === 正式计算 ===
    print("[itemcf_sim_baseline] 🚧 正在计算相似度矩阵（无加权）...")
    user_item_time_dict = get_user_item_time(df)

    i2i_sim = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for i, i_click_time in item_time_list:
            item_cnt[i] += 1
            i2i_sim.setdefault(i, {})
            for j, j_click_time in item_time_list:
                if i == j:
                    continue
                i2i_sim[i].setdefault(j, 0)
                i2i_sim[i][j] += 1 / math.log(len(item_time_list) + 1)

    i2i_sim_ = i2i_sim.copy()
    for i, related_items in i2i_sim.items():
        for j, wij in related_items.items():
            i2i_sim_[i][j] = wij / math.sqrt(item_cnt[i] * item_cnt[j])

    # === 保存缓存 ===
    with open(save_path, 'wb') as f:
        pickle.dump(i2i_sim_, f)
    print(f"[itemcf_sim_baseline] ✅ 相似度矩阵已保存至：{save_path}")

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
    for loc, (i, click_time) in enumerate(user_hist_items):
        for j, wij in sorted(i2i_sim[i].items(), key=lambda x: x[1], reverse=True)[:sim_item_topk]:
            if j in user_hist_items:
                continue

            item_rank.setdefault(j, 0)
            item_rank[j] += wij

    # 不足10个，用热门商品补全
    if len(item_rank) < recall_item_num:
        for i, item in enumerate(item_topk_click):
            if item in item_rank.items():  # 填充的item应该不在原来的列表中
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