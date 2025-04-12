import pandas as pd
from tqdm import tqdm
import os

from data_processing import get_all_click_sample, get_item_info_df
from utils import get_user_item_time, get_item_topk_click, get_item_info_dict
from models.YouTubeDNN_recall import youtubednn_u2i_dict

data_path = './data_raw/'
save_path = 'cache/'

def debug_youtube_dnn_pipeline(sample_nums=1000):
    print("🛠️ Debug 模式启动：YouTubeDNN 双塔召回模型")
    
    # 确保缓存目录存在
    os.makedirs(save_path, exist_ok=True)

    # Step 1：采样用户点击数据
    all_click_df = get_all_click_sample(data_path, sample_nums=sample_nums)
    print(f"✅ Step 1：采样完成，共 {len(all_click_df)} 条点击记录")

    # Step 2：加载文章信息
    item_info_df = get_item_info_df(data_path)
    print(f"✅ Step 2：加载文章信息完成，共 {len(item_info_df)} 篇文章")

    # Step 3：训练YouTubeDNN模型并生成召回结果
    print("🚀 Step 3：开始训练YouTubeDNN模型并生成召回结果")
    user_recall_items_dict = youtubednn_u2i_dict(
        data=all_click_df,
        save_path=save_path,
        topk=10,          # 为每个用户召回10篇文章
        epochs=2,         # 训练2轮
        batch_size=256,   # 批大小为256
        validation_split=0.1  # 10%的数据作为验证集
    )

    # Step 4：展示召回结果
    print("\n📊 YouTubeDNN召回结果：")
    user_count = 0
    for user, rec_list in user_recall_items_dict.items():
        print(f"\n👤 用户 {user} 的推荐结果：")
        for item, score in rec_list:
            print(f"  📄 文章 {item}, 相似度得分 {score:.4f}")
        user_count += 1
        if user_count >= 5:  # 只展示5个用户的结果
            break

    print("\n🎉 YouTubeDNN Debug 流程完成 ✅")


if __name__ == '__main__':
    debug_youtube_dnn_pipeline(sample_nums=3000)  # 使用3000条样本，确保有足够的训练数据 