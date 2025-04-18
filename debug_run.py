import pandas as pd
from tqdm import tqdm

from data_processing import get_all_click_sample, get_item_info_df
from utils import get_user_item_time, get_item_topk_click, get_item_info_dict
from models.itemcf_recall import itemcf_sim, generate_itemcf_recall_dict  # 最新加权召回
from data_processing import embdding_sim

data_path = './data_raw/'

def debug_pipeline(sample_nums=1000, emb_sample_n=500):
    print("🛠️ Debug 模式启动：当前使用加权 ItemCF + Embedding")

    # Step 1：采样用户点击数据
    all_click_df = get_all_click_sample(data_path, sample_nums=sample_nums)
    print(f"✅ Step 1：采样完成，共 {len(all_click_df)} 条点击记录")

    # Step 2：加载文章信息并构建相关字典
    item_info_df = get_item_info_df(data_path)
    item_type_dict, item_words_dict, item_created_time_dict = get_item_info_dict(item_info_df)

    # Step 3：加载并抽样 embedding，构建 embedding 相似度
    print("🚀 Step 3：加载文章 embedding 并抽样构建相似度")
    # 不修改 all_click_df，仅用于 embedding 相似度构建
    item_emb_df = pd.read_csv(data_path + '/articles_emb.csv').sample(n=emb_sample_n, random_state=42)
    emb_item_ids = set(item_emb_df['article_id'])
    click_df_for_emb = all_click_df[all_click_df['click_article_id'].isin(emb_item_ids)]

    # embedding_sim 只用 click_df_for_emb，而不要污染主流程的 all_click_df
    emb_i2i_sim = embdding_sim(click_df_for_emb, item_emb_df, save_path='cache/', topk=10)
    print(f"✅ Step 3：完成 embedding 相似度计算")

    # Step 4：计算 ItemCF 相似度
    i2i_sim = itemcf_sim(all_click_df, item_created_time_dict, save_path='cache/', use_cache=False)
    print(f"✅ Step 4：ItemCF 相似度计算完成，共 {len(i2i_sim)} 个物品")

    # Step 5：构建召回
    val_df = pd.DataFrame({'user_id': all_click_df['user_id'].unique()[:5]})
    user_item_time_dict = get_user_item_time(all_click_df)
    item_topk_click = get_item_topk_click(all_click_df, k=50)

    user_recall_items_dict = generate_itemcf_recall_dict(
        val_df=val_df,
        user_item_time_dict=user_item_time_dict,
        i2i_sim=i2i_sim,
        sim_item_topk=10,
        recall_item_num=10,
        item_topk_click=item_topk_click,
        item_created_time_dict=item_created_time_dict,
        emb_i2i_sim=emb_i2i_sim,
        save_path='cache/user_recall_debug.pkl',
        use_cache=False
    )

    for user, rec_list in user_recall_items_dict.items():
        print(f"\n👤 用户 {user} 的推荐结果：")
        for item, score in rec_list:
            print(f"  📄 文章 {item}, 相似度得分 {score:.4f}")

    print("\n🎉 Debug 流程完成 ✅")



if __name__ == '__main__':
    debug_pipeline(sample_nums=1000)