import os
from tqdm import tqdm
import pickle
import pandas as pd
import warnings
from collections import defaultdict
import collections
warnings.filterwarnings('ignore')


from data_processing import get_all_click_df
from data_processing import get_user_item_time
from recall import itemcf_sim
from recall import item_based_recommend
from data_processing import get_item_topk_click
from evaluation import submit
from evaluation import calculate_mrr

data_path = './data_raw/'
save_path = './results/'

def main():
    print("🚀 开始执行 main()...")

    # 全量训练集
    print("📌 [Step 1] 读取全量训练集...")
    all_click_df = get_all_click_df(offline=False)
    print(f"✅ 训练集加载完成，共 {len(all_click_df)} 条点击记录")

    # 构建物品相似度矩阵
    print("📌 [Step 2] 计算物品相似度矩阵...")
    sim_matrix_file = save_path + 'itemcf_i2i_sim.pkl'

    if os.path.exists(sim_matrix_file):
        print("✅ 发现已有相似度矩阵，正在加载...")
        i2i_sim = pickle.load(open(sim_matrix_file, 'rb'))
        print("✅ 相似度矩阵加载完成")
    else:
        print("📌 未发现相似度矩阵，开始计算...")
        i2i_sim = itemcf_sim(all_click_df, save_path)
        print("✅ 相似度矩阵计算完成，并已保存")

    # 定义召回字典
    print("📌 [Step 3] 初始化用户召回字典...")
    user_recall_items_dict = collections.defaultdict(dict)

    # 获取 用户 - 文章 - 点击时间的字典
    print("📌 [Step 4] 计算用户-文章点击时间序列...")
    user_item_time_dict = get_user_item_time(all_click_df)
    print(f"✅ 用户-文章时间序列计算完成，共 {len(user_item_time_dict)} 个用户")

    # 去取文章相似度
    print("📌 [Step 5] 读取物品相似度矩阵...")
    i2i_sim = pickle.load(open(save_path + 'itemcf_i2i_sim.pkl', 'rb'))
    print("✅ 物品相似度矩阵加载成功")

    # 相似文章的数量
    sim_item_topk = 10
    print(f"📌 [Step 6] 设置相似物品数量为 {sim_item_topk}")

    # 召回文章数量
    recall_item_num = 10
    print(f"📌 [Step 7] 设定召回文章数量为 {recall_item_num}")

    # 用户热度补全
    print("📌 [Step 8] 计算最热文章...")
    item_topk_click = get_item_topk_click(all_click_df, k=50)
    print(f"✅ 热门文章计算完成，前5篇为 {item_topk_click[:5]}")

    print("📌 [Step 9] 开始为每个用户生成推荐列表...")
    # 定义推荐结果文件路径
    user_recall_file = save_path + 'user_recall_items.pkl'

    if os.path.exists(user_recall_file):
        print("✅ 发现已有用户推荐列表，正在加载...")
        user_recall_items_dict = pickle.load(open(user_recall_file, 'rb'))
        print("✅ 用户推荐列表加载完成")
    else:
        print("📌 未发现用户推荐列表，开始计算...")

        user_recall_items_dict = {}
        for idx, user in enumerate(tqdm(all_click_df['user_id'].unique())):
            user_recall_items_dict[user] = item_based_recommend(user, user_item_time_dict, i2i_sim,
                                                                sim_item_topk, recall_item_num, item_topk_click)
            if idx % 10000 == 0:
                print(f"✅  已处理 {idx}/{len(all_click_df['user_id'].unique())} 个用户...")

        print("✅ 所有用户的推荐列表生成完成")

        # 保存计算结果
        pickle.dump(user_recall_items_dict, open(user_recall_file, 'wb'))
        print("✅ 用户推荐列表已保存，后续可直接加载")

    # 将字典的形式转换成df
    print("📌 [Step 10] 将推荐字典转换为 DataFrame...")
    user_item_score_list = []

    for user, items in tqdm(user_recall_items_dict.items()):
        for item, score in items:
            user_item_score_list.append([user, item, score])

    recall_df = pd.DataFrame(user_item_score_list, columns=['user_id', 'click_article_id', 'pred_score'])
    print(f"✅ 推荐 DataFrame 生成完成，共 {len(recall_df)} 条数据")

    # 获取测试集
    print("📌 [Step 11] 读取测试集...")
    tst_click = pd.read_csv(data_path + 'testA_click_log.csv')
    tst_users = tst_click['user_id'].unique()
    print(f"✅ 测试集加载完成，共 {len(tst_users)} 个用户")

    # 从所有的召回数据中将测试集中的用户选出来
    print("📌 [Step 12] 过滤测试集用户的推荐数据...")
    tst_recall = recall_df[recall_df['user_id'].isin(tst_users)]
    print(f"✅ 过滤完成，共 {len(tst_recall)} 条推荐数据")

    # 生成提交文件
    print("📌 [Step 13] 生成提交文件...")
    submit(tst_recall, save_path, topk=5, model_name='itemcf_baseline')
    print("✅ 提交文件已生成！")

    # 打分
    print("📌 [Step 14] 打分...")
    calculate_mrr(data_path + 'testA_click_log.csv', save_path + 'itemcf_baseline_03-17.csv')

    print("🎉 main() 运行完成！")

if __name__ == "__main__":
    main()