import numpy as np

# 依次评估召回的前10, 20, 30, 40, 50个文章中的击中率
def metrics_recall(user_recall_items_dict, val_df, topk=None):
    """
    召回指标的评估 - 检查召回的物品中是否包含用户在验证集中交互的物品
    """
    # 获取验证集中用户的最后一次点击
    val_user_items = dict(zip(val_df['user_id'], val_df['click_article_id']))
    
    # 计算召回覆盖率
    covered_users = set(user_recall_items_dict.keys()) & set(val_user_items.keys())
    coverage = len(covered_users) / len(val_user_items)
    print(f"📊 用户覆盖率: {coverage:.4f} ({len(covered_users)}/{len(val_user_items)})")
    
    # 多层次评估
    if topk is None:
        for k in [10, 20, 30, 40, 50]:
            hit = 0
            for user in covered_users:
                true_item = val_user_items[user]
                recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
                if true_item in recall_items:
                    hit += 1
            
            recall = round(hit / len(covered_users), 5) if covered_users else 0
            print(f"📊 Recall@{k}: {recall:.5f} ({hit}/{len(covered_users)})")
            
            # 计算前10个物品的平均分数，判断分数分布
            if k == 10:
                avg_scores = []
                for user in list(covered_users)[:20]:
                    scores = [score for _, score in user_recall_items_dict[user][:10]]
                    if scores:
                        avg_scores.append(np.mean(scores))
                if avg_scores:
                    print(f"📊 前10个物品的平均分数: {np.mean(avg_scores):.4f}")
        
        # 计算最大的k
        k = 50
        hit = 0
        for user in covered_users:
            true_item = val_user_items[user]
            recall_items = [x[0] for x in user_recall_items_dict[user][:k]]
            if true_item in recall_items:
                hit += 1
        
        recall = round(hit / len(covered_users), 5) if covered_users else 0
        return recall
    
    # 单层评估
    else:
        hit = 0
        for user in covered_users:
            true_item = val_user_items[user]
            recall_items = [x[0] for x in user_recall_items_dict[user][:topk]]
            if true_item in recall_items:
                hit += 1
        
        recall = round(hit / len(covered_users), 5) if covered_users else 0
        print(f"📊 Recall@{topk}: {recall:.5f} ({hit}/{len(covered_users)})")
        return recall
