import json
import csv
import nltk
from collections import defaultdict
import math

# 引入我们在阶段三逆向改造时暴露的纯净推理接口
from extractor_api import YelpFeatureExtractor


def execute_phase_5_pipeline(review_json_path, mapping_csv_path, output_feature_path, batch_size=64, limit=5000):
    print("启动阶段五：大规模推理与特征对齐流水线...\n")

    # =====================================================================
    # Step 1: 抽取基座的热加载与状态锁定 (Warm Loading & State Locking)
    # =====================================================================
    print("[1/5] 执行热加载与状态锁定...")
    # 实例化阶段三的纯净接口，底层已默认调用 model.eval() 和 torch.no_grad()
    extractor = YelpFeatureExtractor(domain_config_name="yelp_restaurant")
    print("      => 抽取引擎挂载完毕，计算图已安全冻结。")

    # 预先加载映射表到内存 (O(1) 查询复杂度)
    review_to_poi = {}
    with open(mapping_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            if len(row) == 2:
                review_to_poi[row[0]] = row[1]

    # 用于暂存 Review 级别和 POI 级别的特征字典
    review_features = defaultdict(lambda: defaultdict(int))

    # =====================================================================
    # Step 2: 实体降维与微观粒度切分 (Entity Dimensionality Reduction & Segmentation)
    # =====================================================================
    print("\n[2/5] 开始流式实体降维与微观粒度切分...")
    sentence_buffer = []
    cursor_buffer = []  # 隐形游标，用于记录每个句子属于哪个 Review
    processed_reviews = 0

    with open(review_json_path, 'r', encoding='utf-8') as rf:
        for line in rf:
            r_data = json.loads(line)
            r_id = r_data['review_id']

            # 仅处理我们阶段一映射表里存在的合法 Review
            if r_id not in review_to_poi:
                continue

            text = r_data.get('text', '').replace('\n', ' ').strip()

            # 工业级微观切分：将长篇 Review 切分为独立单句
            sentences = nltk.tokenize.sent_tokenize(text)

            for sent in sentences:
                # 撤销 len(sent.split()) > 3 的粗暴拦截
                # 改为只拦截纯标点或极其空洞的无意义字符组合 (比如 "...", "ok.")
                if len(sent.strip()) >= 4:
                    sentence_buffer.append(sent)
                    cursor_buffer.append(r_id)  # 挂载游标

            processed_reviews += 1
            if processed_reviews >= limit:
                break

    print(f"      => 成功将 {processed_reviews} 篇评论降维拆解为 {len(sentence_buffer)} 个独立单句。")

    # =====================================================================
    # Step 3: 高并发流式张量推理 (High-Concurrency Streaming Tensor Inference)
    # =====================================================================
    print("\n[3/5] 启动动态批处理与软投票特征分配...")
    total_batches = math.ceil(len(sentence_buffer) / batch_size)

    # 【新增】定义双重置信度阈值 (垃圾桶的拦截线)
    ASPECT_THRESHOLD = 0.45  # 4分类，随机线是0.25
    POLARITY_THRESHOLD = 0.65  # 2分类，随机线是0.50

    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(sentence_buffer))

        batch_sentences = sentence_buffer[start_idx:end_idx]
        batch_cursors = cursor_buffer[start_idx:end_idx]

        # 调用软投票接口
        batch_probs = extractor.extract_sentences_soft(batch_sentences)

        for cursor, probs, sentence in zip(batch_cursors, batch_probs, batch_sentences):
            aspect_probs = probs['aspects']
            polarity_probs = probs['polarities']

            # --- 垃圾桶机制核心逻辑 ---
            # 1. 获取最高维度的概率值
            max_aspect_prob = max(aspect_probs.values())
            max_polarity_prob = max(polarity_probs.values())

            # 2. 双重拦截：只要有一个低于阈值，直接扔进垃圾桶 (continue)
            if max_aspect_prob < ASPECT_THRESHOLD or max_polarity_prob < POLARITY_THRESHOLD:
                # 这句话被判定为无意义的背景噪音或中性陈述，拒绝让它污染特征库
                continue

                # 3. 通过了垃圾桶校验，才允许进行特征分配
            for aspect_name, p_asp in aspect_probs.items():
                if p_asp > 0.1:  # 忽略微小概率的弥散
                    for polarity_name, p_pol in polarity_probs.items():
                        feature_key = f"{aspect_name}_{polarity_name}"
                        joint_prob = p_asp * p_pol

                        review_features[cursor][feature_key] += joint_prob

        if (i + 1) % 10 == 0 or (i + 1) == total_batches:
            print(f"      - 推理进度: {i + 1} / {total_batches} Batches 完成")

    # =====================================================================
    # Step 4: 跨层级多维特征聚合 (Cross-level Multi-dimensional Feature Aggregation)
    # =====================================================================
    print("\n[4/5] 执行跨层级多维特征聚合 (Sentence -> Review -> POI)...")
    poi_features = defaultdict(lambda: defaultdict(int))

    for r_id, features in review_features.items():
        b_id = review_to_poi[r_id]  # 召唤映射表
        # 将该 Review 的所有情感得分汇聚到它所属的 POI (商户) 节点上
        for feat_key, count in features.items():
            poi_features[b_id][feat_key] += count

    print(f"      => 成功将零散特征汇聚至 {len(poi_features)} 个独特的 POI 节点。")

    # =====================================================================
    # Step 5: 稠密向量生成与推荐网关对接 (Dense Vector Generation & Gateway Docking)
    # =====================================================================
    print("\n[5/5] 生成含热度保留与平滑归一化的稠密向量...")

    aspects = ['food', 'service', 'ambience', 'price']
    polarities = ['positive', 'negative']
    vector_columns = [f"{a}_{p}" for a in aspects for p in polarities]

    # 【新增】贝叶斯平滑系数 (Laplace Smoothing Factor)
    # 含义：假设每个商户在被计算前，都已经有了 10 句"中性/无效"的评价作为底噪。
    # 作用：防止小样本商户出现 1.0 的极端满分。
    SMOOTHING_FACTOR = 10.0

    with open(output_feature_path, 'w', encoding='utf-8') as out_f:
        # 【新增】表头增加一列 log_volume，专门记录 POI 的全局热度
        header = ["poi_id", "log_volume"] + vector_columns
        out_f.write(",".join(header) + "\n")

        for b_id, features in poi_features.items():
            raw_vector = [features.get(col, 0) for col in vector_columns]
            total_mentions = sum(raw_vector)

            # 【新增】计算并保留热度特征：使用 log1p (即 ln(1 + x)) 防止数值溢出
            # 这个值将告诉推荐系统该商户在 Yelp 上的曝光量级
            log_volume = round(math.log1p(total_mentions), 4)

            # 【修改】使用贝叶斯平滑进行归一化
            if total_mentions == 0:
                normalized_vector = [0.0] * len(vector_columns)
            else:
                # 分母加上 SMOOTHING_FACTOR，惩罚低频冷门商户
                smoothed_denominator = total_mentions + SMOOTHING_FACTOR
                normalized_vector = [round(val / smoothed_denominator, 4) for val in raw_vector]

            # 结构化拼接落盘：主键 + 热度标量 + 情感分布向量
            row_data = [b_id, str(log_volume)] + [str(v) for v in normalized_vector]
            out_f.write(",".join(row_data) + "\n")

    print(f"      => 稠密特征网关文件生成完毕: {output_feature_path}")


if __name__ == "__main__":
    execute_phase_5_pipeline(
        review_json_path="data/yelp/yelp_academic_dataset_review.json",
        mapping_csv_path="data/yelp_restaurant/review_to_poi_mapping.csv",
        output_feature_path="data/yelp_restaurant/poi_dense_features_gateway.csv",
        batch_size=64,
        limit=5000  # 测试阶段限制处理 5000 条 Review
    )