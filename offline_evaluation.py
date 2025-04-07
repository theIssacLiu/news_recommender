import os
import pandas as pd
from utils import (
    get_user_item_time,
    get_item_topk_click,
    get_item_info_dict
)
from data_processing import get_all_click_df, get_item_info_df
# from models.recall import itemcf_sim, generate_user_recall_dict
from recall_evaluation import metrics_recall
from models.recall_baseline import itemcf_sim, generate_user_recall_dict

data_path = './data_raw/'
cache_dir = './cache/offline/'

def split_train_val(all_click_df):
    all_click_df = all_click_df.sort_values(by=['user_id', 'click_timestamp'])
    val_clicks = all_click_df.groupby('user_id').tail(1)
    train_clicks = all_click_df.drop(val_clicks.index)
    return train_clicks, val_clicks

def metrics_mrr(user_recall_items_dict, val_df, topk=5):
    """
    计算所有用户的 MRR@topk（Mean Reciprocal Rank）

    :param user_recall_items_dict: {user_id: [(item_id, score), ...]}
    :param val_df: pd.DataFrame，验证集中每个用户的最后一次点击
    :param topk: int，只评估前 topk 个推荐
    """
    last_click_item_dict = dict(zip(val_df['user_id'], val_df['click_article_id']))
    total_score = 0
    user_cnt = 0

    for user, rec_list in user_recall_items_dict.items():
        if user not in last_click_item_dict:
            continue
        true_item = last_click_item_dict[user]
        mrr_score = 0

        for rank, (item, _) in enumerate(rec_list[:topk], start=1):
            if item == true_item:
                mrr_score += 1 / rank  # 满足 s(user,k)=1 时，加上 1/k
        total_score += mrr_score
        user_cnt += 1

    avg_mrr = round(total_score / user_cnt, 5) if user_cnt else 0.0
    print(f"📊 最终离线 MRR@{topk}：{avg_mrr}")
    return avg_mrr


def offline_evaluate(sim_version='default', recall_version='default', use_cache=True):
    os.makedirs(cache_dir, exist_ok=True)

    # Step 1：加载数据 & 划分训练/验证
    print("📌 加载训练数据并划分验证集...")
    all_click_df = get_all_click_df(data_path=data_path, offline=True)
    train_df, val_df = split_train_val(all_click_df)
    print(f"✅ 训练集 {len(train_df)} 条，验证集 {len(val_df)} 条")

    # Step 2：相似度矩阵（使用封装）
    sim_cache_file = os.path.join(cache_dir, f'itemcf_sim_{sim_version}.pkl')
    item_info_df = get_item_info_df(data_path)
    item_created_time_dict = get_item_info_dict(item_info_df)[2]

    i2i_sim = itemcf_sim(
        train_df,
        item_created_time_dict,
        save_path=sim_cache_file,
        use_cache=use_cache
    )

    # Step 3：用户点击信息 + 热度文章
    user_item_time_dict = get_user_item_time(train_df)
    item_topk_click = get_item_topk_click(train_df, k=50)

    # Step 4：用户召回（使用封装）
    recall_cache_file = os.path.join(cache_dir, f'user_recall_items_{recall_version}.pkl')

    user_recall_items_dict = generate_user_recall_dict(
        val_df,
        user_item_time_dict,
        i2i_sim,
        sim_item_topk=10,
        recall_item_num=50,
        item_topk_click=item_topk_click,
        save_path=recall_cache_file,
        use_cache=use_cache
    )

    # Stp 5：评估最终MRR分数
    print("📊 开始评估召回效果...")
    metrics_recall(user_recall_items_dict, val_df, topk=50)
    metrics_mrr(user_recall_items_dict, val_df, topk=5)  # 只评估前5个，符合你给出的比赛规则
if __name__ == '__main__':
    offline_evaluate(
        sim_version='weighted_offline',
        recall_version='weighted_offline',
        use_cache=True
    )
