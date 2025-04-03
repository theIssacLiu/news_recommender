from datetime import datetime

import pandas as pd


# 生成提交文件
def submit(recall_df,save_path, topk=5, model_name=None):
    recall_df = recall_df.sort_values(by=['user_id', 'pred_score'])
    recall_df['rank'] = recall_df.groupby(['user_id'])['pred_score'].rank(ascending=False, method='first')

    # 判断是不是每个用户都有5篇文章及以上
    tmp = recall_df.groupby('user_id').apply(lambda x: x['rank'].max())
    assert tmp.min() >= topk

    del recall_df['pred_score']
    submit = recall_df[recall_df['rank'] <= topk].set_index(['user_id', 'rank']).unstack(-1).reset_index()

    submit.columns = [int(col) if isinstance(col, int) else col for col in submit.columns.droplevel(0)]
    # 按照提交格式定义列名
    submit = submit.rename(columns={'': 'user_id', 1: 'article_1', 2: 'article_2',
                                    3: 'article_3', 4: 'article_4', 5: 'article_5'})

    save_name = save_path + model_name + '_' + datetime.today().strftime('%m-%d') + '.csv'
    submit.to_csv(save_name, index=False, header=True)

# 计算成绩
# 预处理testA的结果表
# 定义计算公式
# 计算所有结果
def calculate_mrr(testA_path, submission_path):
    testA_click = pd.read_csv(testA_path)
    testA_ground_truth = testA_click.sort_values(by=['click_timestamp']).groupby('user_id').last().reset_index()
    ground_truths = dict(zip(testA_ground_truth['user_id'], testA_ground_truth['click_article_id']))

    print(f"✅ 提取 Ground Truth 完成，共 {len(ground_truths)} 个用户")

    submission_df = pd.read_csv(submission_path)

    predictions = dict(zip(submission_df['user_id'],
                           submission_df[
                               ['article_1', 'article_2', 'article_3', 'article_4', 'article_5']].values.tolist()))

    print(f"✅ 读取预测结果，共 {len(predictions)} 个用户")

    mrr_total = 0
    count = 0

    for user_id, truth in ground_truths.items():
        if user_id in predictions:
            pred_list = predictions[user_id] # 获取该用户的推荐列表（前5篇）
            if truth in pred_list:
                rank = pred_list.index(truth) + 1
                mrr_total += 1 / rank # 贡献 MRR 分数
            count += 1 # 统计有效用户数
    mrr_score = mrr_total / count if count > 0 else 0
    print(f"🔥 最终 MRR 评分: {mrr_score:.4f}")

    return mrr_score
