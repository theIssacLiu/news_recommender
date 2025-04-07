# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, trn_last_click_df, topk=50):
    last_click_item_dict = dict(zip(trn_last_click_df['user_id'], trn_last_click_df['click_article_id']))
    user_num = len(user_recall_items_dict)

    print("\n📊 [召回评估] 不同 TopK 下的 Recall 命中率：")
    for k in range(10, topk + 1, 10):
        hit_num = 0
        for user, rec_items in user_recall_items_dict.items():
            tmp_recall_items = [item for item, score in rec_items[:k]]
            if last_click_item_dict.get(user) in tmp_recall_items:
                hit_num += 1

        hit_rate = round(hit_num / user_num, 5)
        print(f"  Top-{k:>2}  |  Hit: {hit_num:<4}  |  Recall@{k}: {hit_rate:.5f}")
