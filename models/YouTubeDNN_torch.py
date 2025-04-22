import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import os
import random
import collections
import faiss
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 定义数据集类
class YouTubeDNNDataset(Dataset):
    def __init__(self, user_ids, item_seqs, target_items, labels, seq_lens, max_len=30):
        self.user_ids = torch.LongTensor(user_ids)
        self.target_items = torch.LongTensor(target_items)
        self.labels = torch.FloatTensor(labels)
        self.seq_lens = torch.LongTensor(np.minimum(seq_lens, max_len))  # 确保不超过max_len
        self.max_len = max_len
        
        # 预处理所有序列，确保长度一致
        self.padded_seqs = []
        for seq in item_seqs:
            if len(seq) > max_len:
                # 截断过长的序列
                padded_seq = seq[:max_len]
            else:
                # 填充过短的序列
                padded_seq = seq + [0] * (max_len - len(seq))
            self.padded_seqs.append(padded_seq)
        self.padded_seqs = torch.LongTensor(self.padded_seqs)
        
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'hist_item_seq': self.padded_seqs[idx],
            'target_item': self.target_items[idx],
            'seq_len': self.seq_lens[idx],
            'label': self.labels[idx]
        }

# 双塔模型定义
class YouTubeDNNModel(nn.Module):
    def __init__(self, user_count, item_count, embedding_dim=16, hidden_units=(64, 16), dropout=0.2):
        super(YouTubeDNNModel, self).__init__()
        self.embedding_dim = embedding_dim
        
        # 用户和物品的嵌入层
        self.user_embedding = nn.Embedding(user_count, embedding_dim)
        self.item_embedding = nn.Embedding(item_count, embedding_dim)
        
        # 历史物品序列聚合
        self.hist_embedding = nn.Embedding(item_count, embedding_dim)
        
        # 用户塔深度网络
        layers = []
        input_dim = embedding_dim * 2  # 用户ID嵌入 + 历史序列嵌入
        for unit in hidden_units:
            layers.append(nn.Linear(input_dim, unit))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = unit
        self.user_dnn = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, user_id, hist_item_seq, target_item, seq_len):
        # 用户ID Embedding
        user_emb = self.user_embedding(user_id)  # [B, E]
        
        # 历史物品序列Embedding
        hist_emb = self.hist_embedding(hist_item_seq)  # [B, L, E]
        
        # 计算序列的平均值
        device = user_id.device
        mask = torch.arange(hist_item_seq.size(1), device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        mask = mask.unsqueeze(2).float()
        hist_emb = (hist_emb * mask).sum(dim=1) / seq_len.unsqueeze(1).float().clamp(min=1)  # [B, E]
        
        # 连接用户嵌入和历史物品嵌入
        user_feature = torch.cat([user_emb, hist_emb], dim=1)  # [B, 2E]
        
        # 用户塔DNN
        user_dnn_out = self.user_dnn(user_feature)  # [B, last_hidden]
        
        # 目标物品Embedding
        item_emb = self.item_embedding(target_item)  # [B, E]
        
        # 计算点积
        if len(item_emb.shape) == 3:  # 批量计算多个物品 [B, N, E]
            score = torch.bmm(user_dnn_out.unsqueeze(1), item_emb.transpose(1, 2)).squeeze(1)  # [B, N]
        else:  # 单个物品 [B, E]
            score = torch.sum(user_dnn_out * item_emb, dim=1)  # [B]
        
        return score
    
    def get_user_embedding(self, user_id, hist_item_seq, seq_len):
        user_emb = self.user_embedding(user_id)
        hist_emb = self.hist_embedding(hist_item_seq)
        
        device = user_id.device
        mask = torch.arange(hist_item_seq.size(1), device=device).unsqueeze(0) < seq_len.unsqueeze(1)
        mask = mask.unsqueeze(2).float()
        hist_emb = (hist_emb * mask).sum(dim=1) / seq_len.unsqueeze(1).float().clamp(min=1)
        
        user_feature = torch.cat([user_emb, hist_emb], dim=1)
        user_dnn_out = self.user_dnn(user_feature)
        
        return user_dnn_out
    
    def get_item_embedding(self, item_id):
        return self.item_embedding(item_id)


# 获取双塔召回时的训练验证数据
# negsample指的是通过滑窗构建样本的时候，负样本的数量
def gen_data_set(data, negsample=0, max_hist_len=30, cache_path=None, use_cache=True):
    """
    生成训练集和测试集，支持缓存
    """
    # 如果启用缓存且缓存文件存在，直接加载
    if use_cache and cache_path and os.path.exists(cache_path):
        print(f"✅ 从缓存加载训练集和测试集: {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("🚀 开始生成训练集和测试集...")
    data.sort_values("click_timestamp", inplace=True)
    
    # 获取编码后的物品ID范围
    item_ids = data['click_article_id_encoded'].unique()
    max_item_id = item_ids.max()
    print(f"编码后物品ID范围: 0-{max_item_id}")
    
    train_set = []
    test_set = []
    
    # 使用tqdm显示进度
    for reviewerID, hist in tqdm(data.groupby('user_id_encoded'), desc="构建训练样本"):
        pos_list = hist['click_article_id_encoded'].tolist()
        
        if negsample > 0:
            # 修改负采样逻辑，确保采样的ID在有效范围内
            valid_item_ids = item_ids[item_ids < max_item_id + 1]
            candidate_set = list(set(valid_item_ids) - set(pos_list))
            
            if len(candidate_set) < negsample:
                # 如果候选集太小，允许重复采样
                neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=True)
            else:
                # 否则不允许重复采样
                neg_list = np.random.choice(candidate_set, size=len(pos_list) * negsample, replace=False)

        if len(pos_list) == 1:
            train_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, 1))
            test_set.append((reviewerID, [pos_list[0]], pos_list[0], 1, 1))
            continue

        for i in range(1, len(pos_list)):
            hist = pos_list[:i]
            hist = hist[-max_hist_len:] if len(hist) > max_hist_len else hist

            if i != len(pos_list) - 1:
                train_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))
                for negi in range(negsample):
                    neg_item = neg_list[i * negsample + negi]
                    train_set.append((reviewerID, hist[::-1], neg_item, 0, len(hist[::-1])))
            else:
                test_set.append((reviewerID, hist[::-1], pos_list[i], 1, len(hist[::-1])))

    # 打乱数据
    random.shuffle(train_set)
    random.shuffle(test_set)
    
    # 验证生成的数据集
    print("验证数据集...")
    max_user_id = data['user_id_encoded'].max()
    max_item_id = data['click_article_id_encoded'].max()
    
    for sample in train_set + test_set:
        user_id, hist_items, target_item, label, seq_len = sample
        assert user_id <= max_user_id, f"用户ID {user_id} 超出范围 {max_user_id}"
        assert target_item <= max_item_id, f"物品ID {target_item} 超出范围 {max_item_id}"
        assert all(item <= max_item_id for item in hist_items), f"历史物品列表包含超出范围的ID"
    
    print(f"✅ 数据集验证通过")
    print(f"训练集大小: {len(train_set)}")
    print(f"测试集大小: {len(test_set)}")
    
    # 如果指定了缓存路径，保存结果
    if cache_path:
        print(f"💾 保存训练集和测试集到缓存: {cache_path}")
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump((train_set, test_set), f)
    
    return train_set, test_set


def train_youtube_dnn(train_dataloader, test_dataloader, model, device, epochs=5, 
                     learning_rate=0.001, weight_decay=1e-6):
    """
    训练YouTubeDNN模型，添加错误处理和数据验证
    """
    print(f"[train_youtube_dnn] 开始训练，设备: {device}")
    
    # 1. 设置CUDA环境变量，帮助调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    try:
        # 2. 确保模型和数据在同一设备上
        model = model.to(device)
        
        # 3. 使用混合精度训练
        scaler = torch.cuda.amp.GradScaler()
        
        # 定义损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # 记录最佳验证损失，用于保存最佳模型
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{epochs}')):
                try:
                    # 4. 在移动数据到GPU前先验证数据
                    user_id = batch['user_id'].long()
                    hist_item_seq = batch['hist_item_seq'].long()
                    target_item = batch['target_item'].long()
                    seq_len = batch['seq_len'].long()
                    label = batch['label'].float()
                    
                    # 打印详细的调试信息
                    if batch_idx == 0:
                        print(f"用户ID范围: {user_id.min()}-{user_id.max()}, 嵌入层范围: {model.user_embedding.num_embeddings}")
                        print(f"物品ID范围: {hist_item_seq.max()}, 嵌入层范围: {model.item_embedding.num_embeddings}")
                    
                    # 严格的范围检查
                    if user_id.max() >= model.user_embedding.num_embeddings:
                        print(f"警告：用户ID {user_id.max()} 超出范围 {model.user_embedding.num_embeddings}")
                        continue
                    if hist_item_seq.max() >= model.item_embedding.num_embeddings:
                        print(f"警告：物品ID {hist_item_seq.max()} 超出范围 {model.item_embedding.num_embeddings}")
                        continue
                    
                    # 6. 移动数据到设备
                    user_id = user_id.to(device)
                    hist_item_seq = hist_item_seq.to(device)
                    target_item = target_item.to(device)
                    seq_len = seq_len.to(device)
                    label = label.to(device)
                    
                    # 7. 使用混合精度训练
                    with torch.cuda.amp.autocast():
                        scores = model(user_id, hist_item_seq, target_item, seq_len)
                        loss = criterion(scores, label)
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    if (batch_idx + 1) % 100 == 0:
                        avg_train_loss = train_loss / (train_batches + 1)
                        print(f"  Batch {batch_idx+1}/{len(train_dataloader)}, "
                              f"Loss: {loss.item():.4f}, "
                              f"Avg Loss: {avg_train_loss:.4f}")
                        
                except RuntimeError as e:
                    print(f"训练批次 {batch_idx} 出错: {str(e)}")
                    torch.cuda.empty_cache()
                    continue
            
            # 计算平均训练损失
            avg_train_loss = train_loss / max(1, train_batches)
            
            # 验证阶段
            print(f"\n开始第 {epoch+1} 轮验证...")
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(tqdm(test_dataloader, desc="验证")):
                    try:
                        # 在当前设备上进行验证
                        user_id = batch['user_id'].long().to(device)
                        hist_item_seq = batch['hist_item_seq'].long().to(device)
                        target_item = batch['target_item'].long().to(device)
                        seq_len = batch['seq_len'].long().to(device)
                        label = batch['label'].float().to(device)
                        
                        scores = model(user_id, hist_item_seq, target_item, seq_len)
                        loss = criterion(scores, label)
                        
                        val_loss += loss.item()
                        val_batches += 1
                        
                        if (batch_idx + 1) % 100 == 0:
                            print(f"  验证批次 {batch_idx+1}/{len(test_dataloader)}, "
                                  f"Loss: {loss.item():.4f}")
                            
                    except Exception as e:
                        print(f"验证批次 {batch_idx} 出错: {str(e)}")
                        continue
            
            # 计算平均验证损失
            avg_val_loss = val_loss / max(1, val_batches)
            
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')
            
            # 保存最佳模型
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                print(f"发现更好的模型，验证损失: {best_val_loss:.4f}")
        
        # 恢复最佳模型状态
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"已恢复最佳模型，验证损失: {best_val_loss:.4f}")
        
    except Exception as e:
        print(f"训练过程发生错误: {str(e)}")
        raise  # 抛出异常以便查看完整的错误堆栈
        
    print("[train_youtube_dnn] 训练完成!")
    return model


def youtubednn_u2i_dict(data, save_path="./cache/", topk=20, epochs=5, batch_size=256, embedding_dim=32):
    # 定义所有缓存文件路径
    model_cache = os.path.join(save_path, 'youtube_model.pth')
    embeddings_cache = os.path.join(save_path, 'youtube_embeddings.pkl')
    cache_path = os.path.join(save_path, 'youtube_u2i_dict.pkl')
    dataset_cache = os.path.join(save_path, 'youtube_dataset.pkl')
    
    print("[youtubednn_u2i_dict] 🚀 开始YouTubeDNN处理...")
    
    # 确保缓存目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 1. 修改ID编码部分
    print("[youtubednn_u2i_dict] 开始ID编码...")
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    # 获取所有可能的物品ID
    all_item_ids = data['click_article_id'].unique()
    max_item_id = max(all_item_ids)
    print(f"[youtubednn_u2i_dict] 原始物品ID范围: 0-{max_item_id}")
    
    # 确保ID从0开始连续
    data['user_id_encoded'] = user_encoder.fit_transform(data['user_id'])
    data['click_article_id_encoded'] = item_encoder.fit_transform(data['click_article_id'])
    
    # 获取编码后的用户和物品数量，并添加一些余量
    user_count = len(user_encoder.classes_) + 1  # +1 for padding
    item_count = len(item_encoder.classes_) + 1  # +1 for padding
    
    # 验证编码结果
    max_encoded_item = data['click_article_id_encoded'].max()
    print(f"[youtubednn_u2i_dict] 编码后 - 用户数量: {user_count}, 物品数量: {item_count}")
    print(f"[youtubednn_u2i_dict] 编码后物品ID最大值: {max_encoded_item}")
    
    # 创建ID映射字典以便调试
    id_mapping = dict(zip(item_encoder.classes_, item_encoder.transform(item_encoder.classes_)))
    
    # 验证所有物品ID都在正确范围内
    invalid_ids = data[data['click_article_id_encoded'] >= item_count]['click_article_id'].unique()
    if len(invalid_ids) > 0:
        print(f"[警告] 发现 {len(invalid_ids)} 个超出范围的物品ID")
        print(f"样例: {invalid_ids[:5]}")
        
    # 创建模型实例 - 使用验证后的item_count
    model = YouTubeDNNModel(
        user_count, 
        item_count,  # 确保这个数值足够大
        embedding_dim=embedding_dim,
        hidden_units=(128, 64, embedding_dim),
        dropout=0.3
    )
    
    # 2. 检查是否存在预训练模型
    if os.path.exists(model_cache):
        print(f"[youtubednn_u2i_dict] ✅ 加载预训练模型：{model_cache}")
        model.load_state_dict(torch.load(model_cache))
    else:
        # 如果没有预训练模型，进行训练
        train_set, test_set = gen_data_set(
            data,  # 这里传入的data已经包含了编码后的ID
            negsample=4, 
            max_hist_len=30,
            cache_path=dataset_cache,
            use_cache=True
        )
        
        # 3. 创建数据集时使用编码后的ID
        train_dataset = YouTubeDNNDataset(
            user_ids=[x[0] for x in train_set],  # 这里已经是编码后的user_id
            item_seqs=[x[1] for x in train_set],  # 这里是编码后的item序列
            target_items=[x[2] for x in train_set],  # 这里是编码后的target_item
            labels=[x[3] for x in train_set],
            seq_lens=[x[4] for x in train_set]
        )
        
        test_dataset = YouTubeDNNDataset(
            user_ids=[x[0] for x in test_set],
            item_seqs=[x[1] for x in test_set],
            target_items=[x[2] for x in test_set],
            labels=[x[3] for x in test_set],
            seq_lens=[x[4] for x in test_set]
        )
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 训练模型
        model = train_youtube_dnn(
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            model=model,
            device=device,
            epochs=epochs,
            learning_rate=0.001,
            weight_decay=1e-6
        )
        
        # 保存训练好的模型
        torch.save(model.state_dict(), model_cache)
    
    # 初始化嵌入变量
    user_embeddings = {}
    item_embeddings = {}
    
    if os.path.exists(embeddings_cache):
        print(f"[youtubednn_u2i_dict] ✅ 发现嵌入缓存: {embeddings_cache}")
        with open(embeddings_cache, 'rb') as f:
            cache_data = pickle.load(f)
            user_embeddings = cache_data['user_embeddings']
            item_embeddings = cache_data['item_embeddings']
    else:
        print("[youtubednn_u2i_dict] ⚠️ 未找到嵌入缓存，将重新计算嵌入")
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[youtubednn_u2i_dict] 使用设备: {device}")

        # 将模型移到正确的设备上
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            # 为所有物品计算嵌入
            encoded_items = torch.LongTensor(range(item_count)).to(device)  # 移动到同一设备
            item_embs = model.get_item_embedding(encoded_items).detach().cpu().numpy()
            
            # 保存时使用原始ID
            for idx, orig_item_id in enumerate(item_encoder.classes_):
                item_embeddings[orig_item_id] = item_embs[idx]
            
            # 为所有用户计算嵌入
            for user_id in tqdm(data['user_id'].unique(), desc="计算用户嵌入"):
                # 获取用户的历史交互
                user_hist = data[data['user_id'] == user_id]['click_article_id_encoded'].tolist()
                if not user_hist:
                    continue
                
                # 准备模型输入
                encoded_user_id = user_encoder.transform([user_id])[0]
                hist_items = user_hist[-30:]  # 最多使用最近30个交互
                hist_len = len(hist_items)
                hist_tensor = torch.LongTensor(hist_items + [0] * (30 - hist_len)).to(device)  # 移动到同一设备
                hist_tensor = hist_tensor.unsqueeze(0)
                user_tensor = torch.LongTensor([encoded_user_id]).to(device)  # 移动到同一设备
                seq_len = torch.LongTensor([hist_len]).to(device)  # 移动到同一设备
                
                # 获取用户嵌入
                try:
                    user_emb = model.get_user_embedding(user_tensor, hist_tensor, seq_len).cpu().numpy()  # 先转到CPU
                    user_embeddings[user_id] = user_emb.squeeze() / np.linalg.norm(user_emb)
                except Exception as e:
                    print(f"[youtubednn_u2i_dict] ⚠️ 处理用户 {user_id} 嵌入时出错: {str(e)}")
                    continue
        
        # 保存嵌入
        cache_data = {
            'user_embeddings': user_embeddings,
            'item_embeddings': item_embeddings,
            'user_encoder': user_encoder,
            'item_encoder': item_encoder
        }
        with open(embeddings_cache, 'wb') as f:
            pickle.dump(cache_data, f)
    
    # 使用Faiss进行向量检索
    print("[youtubednn_u2i_dict] 使用Faiss进行向量检索...")
    user_ids = list(user_embeddings.keys())  # 使用user_embeddings而不是user_embs
    user_embs = np.array([user_embeddings[user_id] for user_id in user_ids], dtype=np.float32)
    
    item_ids = list(item_embeddings.keys())  # 使用item_embeddings而不是item_embs
    item_embs = np.array([item_embeddings[item_id] for item_id in item_ids], dtype=np.float32)
    
    # 构建索引
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(item_embs)
    
    # 搜索最相似的物品
    sim, idx = index.search(user_embs, topk)
    
    # 生成召回结果
    user_recall_items_dict = {}
    for i, user_id in enumerate(user_ids):
        item_list = []
        for j, item_idx in enumerate(idx[i]):
            if item_idx < len(item_ids):  # 防止索引越界
                item_id = item_ids[item_idx]
                score = sim[i][j]
                item_list.append((item_id, float(score)))
        user_recall_items_dict[user_id] = item_list
    
    # 保存召回结果
    with open(cache_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)
    
    print(f"[youtubednn_u2i_dict] ✅ 召回结果已保存至: {cache_path}")
    
    # 修改检查代码部分
    print("[youtubednn_u2i_dict] 检查嵌入质量...")
    with torch.no_grad():
        # 抽样检查一些用户嵌入和物品嵌入的余弦相似度
        # 从字典中抽样，而不是从numpy数组中抽样
        user_sample = list(user_embeddings.items())[:3]
        item_sample = list(item_embeddings.items())[:5]
        
        print("样本用户嵌入:")
        for u_id, u_emb in user_sample:
            print(f"用户ID: {u_id}, 嵌入范数: {np.linalg.norm(u_emb)}")
            
            # 检查与样本物品的相似度
            for i_id, i_emb in item_sample:
                sim = np.dot(u_emb, i_emb) / (np.linalg.norm(u_emb) * np.linalg.norm(i_emb))
                print(f"  与物品 {i_id} 的相似度: {sim:.4f}")
    
    return user_recall_items_dict 

def get_youtube_recall(train_df, val_df, save_path, use_cache=True, epochs=10, 
                      batch_size=32,  # 减小batch_size
                      embedding_dim=32):
    """
    使用PyTorch版本的YouTubeDNN模型生成用户-物品召回表
    
    Args:
        train_df: 训练数据
        val_df: 验证数据
        save_path: 结果保存路径
        use_cache: 是否使用缓存
        epochs: 训练轮数
        batch_size: 批大小
        embedding_dim: 嵌入维度
        
    Returns:
        用户-物品召回表，格式为{用户ID: [(物品ID, 得分), ...]}
    """
    # 定义相关缓存路径
    cache_path = os.path.join(save_path, 'youtube_u2i_dict.pkl')
    model_path = os.path.join(save_path, 'youtube_dnn_model.pth')
    user_emb_path = os.path.join(save_path, 'user_youtube_emb.pkl') 
    item_emb_path = os.path.join(save_path, 'item_youtube_emb.pkl')
    
    # 确保目录存在
    os.makedirs(save_path, exist_ok=True)
    
    # 检查召回结果缓存
    if use_cache and os.path.exists(cache_path):
        print(f"[get_youtube_recall] ✅ 使用缓存：{cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("[get_youtube_recall] 🚀 生成YouTubeDNN召回结果...")
    
    # 确保所有ID都经过编码
    item_encoder = LabelEncoder()
    user_encoder = LabelEncoder()
    
    # 先对所有可能的ID进行fit
    all_item_ids = pd.concat([train_df['click_article_id'], val_df['click_article_id']]).unique()
    all_user_ids = pd.concat([train_df['user_id'], val_df['user_id']]).unique()
    
    item_encoder.fit(all_item_ids)
    user_encoder.fit(all_user_ids)
    
    # 然后再transform
    train_df['click_article_id_encoded'] = item_encoder.transform(train_df['click_article_id'])
    val_df['click_article_id_encoded'] = item_encoder.transform(val_df['click_article_id'])
    
    # 仅使用用户和物品ID，简化处理
    df = pd.concat([train_df, val_df], ignore_index=True)
    user_count = df['user_id'].nunique() + 1  # +1 避免索引越界
    item_count = df['click_article_id'].nunique() + 1  # +1 避免索引越界
    
    # 获取所有唯一用户和物品ID
    unique_users = df['user_id'].unique()
    unique_items = df['click_article_id'].unique()
    
    # 获取用户历史交互
    user_hist_dict = {}
    for user_id, group in df.groupby('user_id'):
        user_hist_dict[user_id] = group.sort_values('click_timestamp')['click_article_id'].tolist()
    
    # 创建模型时使用较小的隐藏层
    model = YouTubeDNNModel(
        user_count, 
        item_count, 
        embedding_dim=embedding_dim,
        hidden_units=(64, 32),  # 减小隐藏层
        dropout=0.2
    )
    
    # 检查模型缓存
    if use_cache and os.path.exists(model_path):
        print(f"[get_youtube_recall] ✅ 加载预训练模型：{model_path}")
        model.load_state_dict(torch.load(model_path))
    
    # 检查嵌入缓存
    if use_cache and os.path.exists(user_emb_path) and os.path.exists(item_emb_path):
        print(f"[get_youtube_recall] ✅ 加载用户和物品嵌入")
        with open(user_emb_path, 'rb') as f:
            user_embeddings = pickle.load(f)
        with open(item_emb_path, 'rb') as f:
            item_embeddings = pickle.load(f)
    else:
        # 生成用户和物品的嵌入
        print("[get_youtube_recall] 计算用户和物品嵌入...")
        model.eval()
        
        # 为所有物品生成嵌入
        with torch.no_grad():
            all_item_ids = torch.LongTensor(unique_items)
            all_item_embs = model.get_item_embedding(all_item_ids).detach().numpy()
            normalized_item_embs = all_item_embs / np.linalg.norm(all_item_embs, axis=1, keepdims=True)
            
            # 保存物品嵌入字典
            item_embeddings = {item_id: emb for item_id, emb in zip(unique_items, normalized_item_embs)}
            with open(item_emb_path, 'wb') as f:
                pickle.dump(item_embeddings, f)
        
        # 计算用户嵌入
        user_embeddings = {}
        max_seq_len = 30
        
        with torch.no_grad():
            for user_id in tqdm(unique_users, desc="计算用户嵌入"):
                if user_id not in user_hist_dict or len(user_hist_dict[user_id]) == 0:
                    continue
                    
                # 获取历史交互，确保内容在item_count范围内
                hist_items = [i for i in user_hist_dict[user_id] if i < item_count]
                if not hist_items:
                    continue
                    
                # 最多使用最近30个交互
                hist_items = hist_items[-max_seq_len:] if len(hist_items) > max_seq_len else hist_items
                hist_len = len(hist_items)
                
                # 将历史交互转换为模型输入，确保padding正确
                hist_tensor = torch.LongTensor(hist_items + [0] * (max_seq_len - hist_len))
                hist_tensor = hist_tensor.unsqueeze(0)  # 增加批次维度
                user_tensor = torch.LongTensor([user_id])
                seq_len = torch.LongTensor([hist_len])
                
                # 获取用户嵌入
                try:
                    user_emb = model.get_user_embedding(user_tensor, hist_tensor, seq_len).numpy()
                    user_embeddings[user_id] = user_emb.squeeze() / np.linalg.norm(user_emb)
                except Exception as e:
                    print(f"[get_youtube_recall] ⚠️ 处理用户 {user_id} 嵌入时出错: {str(e)}")
                    continue
        
        # 保存用户嵌入
        with open(user_emb_path, 'wb') as f:
            pickle.dump(user_embeddings, f)
    
    # 准备向量检索
    print("[get_youtube_recall] 使用Faiss进行向量检索...")
    user_ids = list(user_embeddings.keys())
    user_embs = np.array([user_embeddings[user_id] for user_id in user_ids], dtype=np.float32)
    
    item_ids = list(item_embeddings.keys())
    item_embs = np.array([item_embeddings[item_id] for item_id in item_ids], dtype=np.float32)
    
    # 构建索引
    index = faiss.IndexFlatIP(embedding_dim)
    index.add(np.ascontiguousarray(item_embs))
    
    # 为验证集中的用户生成召回结果
    user_recall_items_dict = {}
    topk = recall_num  # 每个用户召回指定数量的文章
    
    # 执行向量检索
    sim, idx = index.search(np.ascontiguousarray(user_embs), topk)
    
    # 构建用户召回结果
    for i, user_id in enumerate(user_ids):
        item_list = []
        for j, item_idx in enumerate(idx[i]):
            if item_idx < len(item_ids):
                item_id = item_ids[item_idx]
                score = float(sim[i][j])
                item_list.append((item_id, score))
        user_recall_items_dict[user_id] = item_list
    
    # 保存召回结果
    with open(cache_path, 'wb') as f:
        pickle.dump(user_recall_items_dict, f)
    
    print(f"[get_youtube_recall] ✅ 召回结果已保存至：{cache_path}")
    return user_recall_items_dict 