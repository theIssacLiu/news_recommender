import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import random
import collections
import faiss
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Flatten, Concatenate, Layer
from tensorflow.keras import regularizers
import tensorflow as tf

# 定义特征列类
class SparseFeat(object):
    def __init__(self, name, vocabulary_size, embedding_dim=8, use_hash=False, dtype='int32', embedding_name=None):
        self.name = name
        self.vocabulary_size = vocabulary_size
        self.embedding_dim = embedding_dim
        self.use_hash = use_hash
        self.dtype = dtype
        self.embedding_name = embedding_name if embedding_name else name

class VarLenSparseFeat(object):
    def __init__(self, sparsefeat, maxlen, combiner='mean', length_name=None):
        self.sparsefeat = sparsefeat
        self.maxlen = maxlen
        self.combiner = combiner
        self.length_name = length_name

# 定义Sampled Softmax Loss
def sampledsoftmaxloss(y_true, y_pred):
    return y_pred

# YouTubeDNN模型定义
class YoutubeDNN(Model):
    def __init__(self, user_feature_columns, item_feature_columns, user_dnn_hidden_units=(64, 32), 
                 dnn_activation='relu', dnn_use_bias=True, dnn_dropout=0, num_sampled=5, seed=1024, **kwargs):
        super(YoutubeDNN, self).__init__(**kwargs)
        
        self.user_feature_columns = user_feature_columns
        self.item_feature_columns = item_feature_columns
        self.hidden_units = user_dnn_hidden_units
        self.activation = dnn_activation
        self.use_bias = dnn_use_bias
        self.dropout = dnn_dropout
        self.num_sampled = num_sampled
        self.seed = seed
        
        # 创建用户嵌入字典
        self.user_embedding_dict = {}
        self.history_feature_list = []
        
        for feat in self.user_feature_columns:
            if isinstance(feat, SparseFeat):
                self.user_embedding_dict[feat.embedding_name] = Embedding(
                    input_dim=feat.vocabulary_size,
                    output_dim=feat.embedding_dim,
                    embeddings_initializer='random_normal',
                    embeddings_regularizer=regularizers.l2(1e-6),
                    name='emb_' + feat.name)
            elif isinstance(feat, VarLenSparseFeat):
                self.user_embedding_dict[feat.sparsefeat.embedding_name] = Embedding(
                    input_dim=feat.sparsefeat.vocabulary_size,
                    output_dim=feat.sparsefeat.embedding_dim,
                    embeddings_initializer='random_normal',
                    embeddings_regularizer=regularizers.l2(1e-6),
                    name='emb_' + feat.sparsefeat.name,
                    mask_zero=True)
                self.history_feature_list.append(feat.name)
        
        # 创建物品嵌入字典
        self.item_embedding_dict = {}
        for feat in self.item_feature_columns:
            if isinstance(feat, SparseFeat):
                self.item_embedding_dict[feat.embedding_name] = Embedding(
                    input_dim=feat.vocabulary_size,
                    output_dim=feat.embedding_dim,
                    embeddings_initializer='random_normal',
                    embeddings_regularizer=regularizers.l2(1e-6),
                    name='emb_' + feat.name)
        
        # 用户塔DNN层
        self.user_dnn_layers = []
        for unit in self.hidden_units:
            self.user_dnn_layers.append(Dense(unit, activation=self.activation))
            self.user_dnn_layers.append(Dropout(self.dropout))
        
        # 输出层
        self.output_layer = Dense(1, use_bias=False)
    
    def call(self, inputs, training=None):
        # 用户特征处理
        user_embeddings = []
        sparse_embedding_list = []
        seq_embedding_list = []
        
        for feat in self.user_feature_columns:
            if isinstance(feat, SparseFeat):
                user_embedding = self.user_embedding_dict[feat.embedding_name](inputs[feat.name])
                user_embeddings.append(user_embedding)
            elif isinstance(feat, VarLenSparseFeat):
                seq_embedding = self.user_embedding_dict[feat.sparsefeat.embedding_name](inputs[feat.name])
                # 计算序列特征的平均值
                hist_len = inputs[feat.length_name]
                mask = tf.sequence_mask(hist_len, feat.maxlen)
                mask = tf.cast(mask, tf.float32)
                mask = tf.expand_dims(mask, axis=-1)
                # 长度归一化
                seq_embedding = seq_embedding * mask
                hist_len = tf.cast(hist_len, tf.float32)
                hist_len = tf.expand_dims(hist_len, axis=-1)
                seq_embedding = tf.reduce_sum(seq_embedding, axis=1) / tf.maximum(hist_len, 1.0)
                user_embeddings.append(seq_embedding)
        
        # 物品特征处理
        item_embeddings = []
        for feat in self.item_feature_columns:
            if isinstance(feat, SparseFeat):
                item_embedding = self.item_embedding_dict[feat.embedding_name](inputs[feat.name])
                item_embeddings.append(item_embedding)
        
        # 处理用户特征向量
        user_embedding = tf.concat(user_embeddings, axis=-1)
        for layer in self.user_dnn_layers:
            user_embedding = layer(user_embedding)
        self.user_embedding = user_embedding
        
        # 处理物品特征向量
        item_embedding = tf.concat(item_embeddings, axis=-1)
        self.item_embedding = item_embedding
        
        # 模型输出
        output = tf.matmul(user_embedding, item_embedding, transpose_b=True)
        self.logits = output
        
        return output
    
    def summary(self):
        user_inputs = {feat.name: Input(shape=(1,) if isinstance(feat, SparseFeat) else (feat.maxlen,), name=feat.name) 
                      for feat in self.user_feature_columns}
        item_inputs = {feat.name: Input(shape=(1,), name=feat.name) for feat in self.item_feature_columns}
        
        all_inputs = {**user_inputs, **item_inputs}
        self.user_input = all_inputs
        self.item_input = item_inputs
        
        Model(inputs=all_inputs, outputs=self.call(all_inputs)).summary()

# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0):
    data.sort_values("click_timestamp", inplace=True)
    item_ids = data['click_article_id'].unique()

    train_set = []
    test_set = []
    for reviewerID, hist in tqdm(data.groupby('user_id')):
        pos_list = hist['click_article_id'].tolist()

        if negsample > 0:
            candidate_set = list(set(item_ids) - set(pos_list))  # 用户没看过的文章里面选择负样本
            neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)  # 对于每个正样本，选择n个负样本

        # 长度只有一个的时候，需要把这条数据也放到训练集中，不然的话最终学到的embedding就会有缺失
        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, len(pos_list)))

        # 滑窗构造正负样本
        for i in range(1, len(pos_list)):
            hist = pos_list[:i]

            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1,
                                  len(hist[::-1])))  # 正样本 [user_id, his_item, pos_item, label, len(his_item)]
                for negi in range(negsample):
                    train_set.append((reviewerID, hist[::-1], neg_list[i * negsample + negi], 0,
                                      len(hist[::-1])))  # 负样本 [user_id, his_item, neg_item, label, len(his_item)]
            else:
                # 将最长的那一个序列长度作为测试数据
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))

    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


# 将输入的数据进行padding，使得序列特征的长度都一致
def gen_model_input(train_set, user_profile, seq_max_len):
    train_uid = np.array([line[0] for line in train_set])
    train_seq = [line[1] for line in train_set]
    train_iid = np.array([line[2] for line in train_set])
    train_label = np.array([line[3] for line in train_set])
    train_hist_len = np.array([line[4] for line in train_set])

    train_seq_pad = pad_sequences(train_seq, maxlen=seq_max_len, padding='post', truncating='post', value=0)
    train_model_input = {"user_id": train_uid, "click_article_id": train_iid, "hist_article_id": train_seq_pad,
                         "hist_len": train_hist_len}

    return train_model_input, train_label


def youtubednn_u2i_dict(data, save_path="./cache/", topk=20, epochs=5, batch_size=256, validation_split=0.1):
    """
    训练YouTubeDNN模型并生成用户-物品召回表
    
    Args:
        data: 用户点击数据，包含user_id、click_article_id和click_timestamp列
        save_path: 结果保存路径
        topk: 为每个用户召回物品的数量
        epochs: 训练轮数
        batch_size: 批大小
        validation_split: 验证集比例
        
    Returns:
        用户-物品召回表，格式为{用户ID: [(物品ID, 得分), ...]}
    """
    # 确保路径存在
    os.makedirs(save_path, exist_ok=True)
    
    # 如果存在缓存直接读取
    cache_path = os.path.join(save_path, 'youtube_u2i_dict.pkl')
    if os.path.exists(cache_path):
        print(f"[youtubednn_u2i_dict] ✅ 使用缓存：{cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("[youtubednn_u2i_dict] 🚀 开始训练YouTubeDNN模型...")
    
    sparse_features = ["click_article_id", "user_id"]
    SEQ_LEN = 30  # 用户点击序列的长度，短的填充，长的截断

    user_profile_ = data[["user_id"]].drop_duplicates('user_id')
    item_profile_ = data[["click_article_id"]].drop_duplicates('click_article_id')

    # 类别编码
    features = ["click_article_id", "user_id"]
    feature_max_idx = {}

    for feature in features:
        lbe = LabelEncoder()
        data[feature] = lbe.fit_transform(data[feature])
        feature_max_idx[feature] = data[feature].max() + 1

    # 提取user和item的画像，这里具体选择哪些特征还需要进一步的分析和考虑
    user_profile = data[["user_id"]].drop_duplicates('user_id')
    item_profile = data[["click_article_id"]].drop_duplicates('click_article_id')

    user_index_2_rawid = dict(zip(user_profile['user_id'], user_profile_['user_id']))
    item_index_2_rawid = dict(zip(item_profile['click_article_id'], item_profile_['click_article_id']))

    # 划分训练和测试集
    # 由于深度学习需要的数据量通常都是非常大的，所以为了保证召回的效果，往往会通过滑窗的形式扩充训练样本
    train_set, test_set = gen_data_set(data, negsample=4)  # 添加负样本
    print(f"[youtubednn_u2i_dict] 训练集样本数：{len(train_set)}，测试集样本数：{len(test_set)}")
    
    # 整理输入数据，具体的操作可以看上面的函数
    train_model_input, train_label = gen_model_input(train_set, user_profile, SEQ_LEN)
    test_model_input, test_label = gen_model_input(test_set, user_profile, SEQ_LEN)

    # 确定Embedding的维度
    embedding_dim = 16

    # 将数据整理成模型可以直接输入的形式
    user_feature_columns = [SparseFeat('user_id', feature_max_idx['user_id'], embedding_dim),
                            VarLenSparseFeat(
                                SparseFeat('hist_article_id', feature_max_idx['click_article_id'], embedding_dim,
                                           embedding_name="click_article_id"), SEQ_LEN, 'mean', 'hist_len'), ]
    item_feature_columns = [SparseFeat('click_article_id', feature_max_idx['click_article_id'], embedding_dim)]

    # 模型的定义
    # num_sampled: 负采样时的样本数量
    model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5,
                       user_dnn_hidden_units=(64, embedding_dim))
    # 模型编译
    model.compile(optimizer="adam", loss=sampledsoftmaxloss)
    model.summary()

    print(f"[youtubednn_u2i_dict] 开始训练，epochs={epochs}, batch_size={batch_size}")
    # 模型训练，这里可以定义验证集的比例，如果设置为0的话就是全量数据直接进行训练
    history = model.fit(train_model_input, train_label, 
                       batch_size=batch_size, 
                       epochs=epochs, 
                       verbose=1, 
                       validation_split=validation_split)

    print("[youtubednn_u2i_dict] 训练完成，提取Embedding...")
    # 训练完模型之后,提取训练的Embedding，包括user端和item端
    test_user_model_input = test_model_input
    all_item_model_input = {"click_article_id": item_profile['click_article_id'].values}

    user_embedding_model = Model(inputs=model.user_input, outputs=model.user_embedding)
    item_embedding_model = Model(inputs=model.item_input, outputs=model.item_embedding)

    # 保存当前的item_embedding 和 user_embedding 排序的时候可能能够用到，但是需要注意保存的时候需要和原始的id对应
    user_embs = user_embedding_model.predict(test_user_model_input, batch_size=2 ** 12)
    item_embs = item_embedding_model.predict(all_item_model_input, batch_size=2 ** 12)

    # embedding保存之前归一化一下
    user_embs = user_embs / np.linalg.norm(user_embs, axis=1, keepdims=True)
    item_embs = item_embs / np.linalg.norm(item_embs, axis=1, keepdims=True)

    # 将Embedding转换成字典的形式方便查询
    raw_user_id_emb_dict = {user_index_2_rawid[k]: \
                                v for k, v in zip(user_profile['user_id'], user_embs)}
    raw_item_id_emb_dict = {item_index_2_rawid[k]: \
                                v for k, v in zip(item_profile['click_article_id'], item_embs)}
    # 将Embedding保存到本地
    pickle.dump(raw_user_id_emb_dict, open(os.path.join(save_path, 'user_youtube_emb.pkl'), 'wb'))
    pickle.dump(raw_item_id_emb_dict, open(os.path.join(save_path, 'item_youtube_emb.pkl'), 'wb'))

    print("[youtubednn_u2i_dict] 使用Faiss进行向量检索...")
    # faiss紧邻搜索，通过user_embedding 搜索与其相似性最高的topk个item
    index = faiss.IndexFlatIP(embedding_dim)
    # 上面已经进行了归一化，这里可以不进行归一化了
    #     faiss.normalize_L2(user_embs)
    #     faiss.normalize_L2(item_embs)
    index.add(item_embs)  # 将item向量构建索引
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)  # 通过user去查询最相似的topk个item

    user_recall_items_dict = collections.defaultdict(dict)
    for target_idx, sim_value_list, rele_idx_list in tqdm(zip(test_user_model_input['user_id'], sim, idx)):
        target_raw_id = user_index_2_rawid[target_idx]
        # 从1开始是为了去掉商品本身, 所以最终获得的相似商品只有topk-1
        for rele_idx, sim_value in zip(rele_idx_list[1:], sim_value_list[1:]):
            rele_raw_id = item_index_2_rawid[rele_idx]
            user_recall_items_dict[target_raw_id][rele_raw_id] = user_recall_items_dict.get(target_raw_id, {}) \
                                                                     .get(rele_raw_id, 0) + sim_value

    user_recall_items_dict = {k: sorted(v.items(), key=lambda x: x[1], reverse=True) for k, v in
                              user_recall_items_dict.items()}
    # 将召回的结果进行排序

    # 保存召回的结果
    # 这里是直接通过向量的方式得到了召回结果，相比于上面的召回方法，上面的只是得到了i2i及u2u的相似性矩阵，还需要进行协同过滤召回才能得到召回结果
    # 可以直接对这个召回结果进行评估，为了方便可以统一写一个评估函数对所有的召回结果进行评估
    pickle.dump(user_recall_items_dict, open(os.path.join(save_path, 'youtube_u2i_dict.pkl'), 'wb'))
    print(f"[youtubednn_u2i_dict] ✅ YouTubeDNN召回结果已保存至：{os.path.join(save_path, 'youtube_u2i_dict.pkl')}")
    
    return user_recall_items_dict