import json
import re
import random
import os


def yelp_phase_one_pipeline(business_json_path, review_json_path, output_txt_path, output_csv_path, max_reviews=10000):
    """
    Yelp 数据预处理与管道设计 - 严格遵循 5 步标准规范
    """
    print(f"Yelp 数据管道，目标抽取数量: {max_reviews} 条\n")

    # =====================================================================
    # Step 1: 确定数据边界与稠密化过滤 (Data Scoping & Filtering)
    # =====================================================================
    print("[1/5] 正在确定数据边界，提取目标类目 (Restaurants) ...")
    valid_businesses = set()

    # 提取目标 POI
    with open(business_json_path, 'r', encoding='utf-8') as bf:
        for line in bf:
            b_data = json.loads(line)
            categories = b_data.get('categories')
            # 数据边界：只保留餐饮类目
            if categories and 'Restaurants' in categories:
                valid_businesses.add(b_data['business_id'])

    print(f"      => 成功圈定 {len(valid_businesses)} 个目标 POI 节点。")

    # 用于校验的内存态映射表
    mapping_keys_for_validation = set()
    processed_count = 0

    print("[2/5] 开始流式清洗评论数据，并建立实体映射表...")

    with open(review_json_path, 'r', encoding='utf-8') as rf, \
            open(output_txt_path, 'w', encoding='utf-8') as out_txt, \
            open(output_csv_path, 'w', encoding='utf-8') as out_csv:

        # 写入 CSV 表头
        out_csv.write("review_id,business_id\n")

        for line in rf:
            r_data = json.loads(line)
            b_id = r_data['business_id']
            r_id = r_data['review_id']
            text = r_data.get('text', '')

            # 稠密化过滤：拦截非目标 POI，拦截单词数少于 15 个字的无意义短评
            if b_id in valid_businesses and len(text.split()) >= 15:

                # =====================================================================
                # Step 3: 严格的文本清洗与防错位规范 (Text Cleaning & Alignment)
                # =====================================================================
                # 【修复点】终极展平处理：利用空参数的 split() 能够自动过滤所有隐藏换行符和 Unicode 分隔符
                cleaned_text = ' '.join(text.split())

                # 防错位规范：剔除行首可能引起框架解析崩溃的不可见字符或特殊非字母数字符号
                cleaned_text = re.sub(r'^[^a-zA-Z0-9]+', '', cleaned_text).strip()

                if not cleaned_text:
                    continue

                # =====================================================================
                # Step 4: 构造目标框架规范的输入流 (Target Format Generation)
                # =====================================================================
                # 严格按照 SBASC 框架要求的格式：正文
                formatted_line = f"{cleaned_text}\n"
                out_txt.write(formatted_line)

                # =====================================================================
                # Step 2: 建立不可篡改的实体映射表 (Entity Mapping Matrix)
                # =====================================================================
                # 外部主键关联：Review ID -> Business ID
                out_csv.write(f"{r_id},{b_id}\n")

                # 记录内存态用于最后的强校验
                mapping_keys_for_validation.add(r_id)
                processed_count += 1

                # 满足局部抽取需求，到达指定数量即安全切断流读取
                if processed_count >= max_reviews:
                    break

    print(f"      => 成功清洗并输出 {processed_count} 条规范化数据。")

    # =====================================================================
    # Step 5: 一致性强校验 (Integrity Validation)
    # =====================================================================
    print("\n[5/5] 启动一致性强校验机制...")

    # 5.1 基数核对 (Cardinality Check)
    txt_line_count = sum(1 for _ in open(output_txt_path, 'r', encoding='utf-8'))
    csv_line_count = sum(1 for _ in open(output_csv_path, 'r', encoding='utf-8')) - 1  # 减去表头

    if txt_line_count != csv_line_count or txt_line_count != len(mapping_keys_for_validation):
        raise RuntimeError(f"致命错误：基数核对失败！文本行数({txt_line_count}) 与 映射表数({csv_line_count}) 不一致！")
    print("      => [通过] 基数核对：100% 一致。")

    # 5.2 逆向抽样探测 (Reverse Sampling Probe)
    sample_size = min(10, txt_line_count)
    with open(output_txt_path, 'r', encoding='utf-8') as check_f:
        lines = check_f.readlines()
        samples = random.sample(lines, sample_size)

        for s_line in samples:
            # 使用正则逆向提取行首的 Review ID
            # 如果你的格式是不带 source 的简写版本
            match = re.match(r'^\[([^\]]+)\]\s', s_line)
            if not match:
                raise ValueError(f"致命错误：目标文件格式被污染，无法解析标准前缀！异常数据: {s_line[:50]}")

            extracted_id = match.group(1)
            if extracted_id not in mapping_keys_for_validation:
                raise KeyError(f"致命错误：文本中提取的 ID {extracted_id} 丢失了 POI 映射！发生严重错位！")

    print(f"      => [通过] 逆向抽样探测：抽取 {sample_size} 条数据防错位验证全部命中。")
    print(f"训练文件: {output_txt_path}")
    print(f"映射表:   {output_csv_path}")


if __name__ == "__main__":
    # 配置你的本地 Yelp 文件路径
    BUSINESS_JSON = "data/yelp/yelp_academic_dataset_business.json"
    REVIEW_JSON = "data/yelp/yelp_academic_dataset_review.json"

    # 输出给 SBASC 框架的文件
    TRAIN_TXT_OUTPUT = "data/yelp_restaurant/train.txt"
    MAPPING_CSV_OUTPUT = "data/yelp_restaurant/review_to_poi_mapping.csv"

    # 仅抽取部分数据跑通代码，这里设置抽取 5000 条
    yelp_phase_one_pipeline(
        business_json_path=BUSINESS_JSON,
        review_json_path=REVIEW_JSON,
        output_txt_path=TRAIN_TXT_OUTPUT,
        output_csv_path=MAPPING_CSV_OUTPUT,
        max_reviews=5000
    )