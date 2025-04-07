import pandas as pd
from tqdm import tqdm

from data_processing import get_all_click_sample
from utils import get_user_item_time, get_item_topk_click
from models.recall import itemcf_sim, item_based_recommend
from data_processing import get_item_info_df
from utils import get_item_info_dict

from models.recall import itemcf_sim  # 加权版
from models.recall_baseline import itemcf_sim_baseline  # baseline版

data_path = './data_raw/'

def debug_pipeline(sample_nums=1000):

    USE_WEIGHTED = False  # 🔁 切换 True or False 来使用不同版本

    print("🛠️ Debug 模式启动：当前使用", "加权ItemCF" if USE_WEIGHTED else "Baseline ItemCF")

    # Step 1：采样一部分用户点击数据
    all_click_df = get_all_click_sample(data_path, sample_nums=sample_nums)
    print(f"✅ 采样完成，共 {len(all_click_df)} 条点击记录，{all_click_df.user_id.nunique()} 个用户")

    # Step 2：过滤其中有交互的文章
    clicked_items = all_click_df['click_article_id'].unique()
    item_info_df = get_item_info_df(data_path)
    item_info_df = item_info_df[item_info_df['click_article_id'].isin(clicked_items)]
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)


    print(f"item_created_time_info 包含 {len(item_info_df)} 篇文章")
    print(f"item_created_time_dict 包含 {len(item_created_time_dict)} 篇文章")
    print(f"点击数据中出现过的文章数：{all_click_df['click_article_id'].nunique()}")


    # Step 3：计算相似度矩阵（不使用缓存，强制重算）

    if USE_WEIGHTED:
        i2i_sim = itemcf_sim(all_click_df, item_created_time_dict, save_path='cache/', use_cache=False)
    else:
        i2i_sim = itemcf_sim_baseline(all_click_df, save_path='cache/', use_cache=False)

    print(f"✅ 相似度矩阵计算完成，包含 {len(i2i_sim)} 个物品")

    # Step 4：召回推荐列表
    user_item_time_dict = get_user_item_time(all_click_df)
    item_topk_click = get_item_topk_click(all_click_df, k=50)

    sim_item_topk = 10
    recall_item_num = 10

    user_recall_items_dict = {}
    for user in tqdm(all_click_df['user_id'].unique()[:5]):  # 只查看前5个用户推荐
        rec_list = item_based_recommend(
            user, user_item_time_dict, i2i_sim,
            sim_item_topk, recall_item_num, item_topk_click
        )
        user_recall_items_dict[user] = rec_list

        print(f"\n👤 用户 {user} 的推荐结果：")
        for item, score in rec_list:
            print(f"  📄 文章 {item}, 相似度得分 {score:.4f}")

    print("\n🎉 Debug 流程完成：相似度计算 + 推荐输出都正常 ✅")


if __name__ == '__main__':
    debug_pipeline(sample_nums=1000)
