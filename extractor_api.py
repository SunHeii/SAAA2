import torch
from hydra import compose, initialize
from models.SBASC.trainer import Trainer
import torch.nn.functional as F

class YelpFeatureExtractor:
    """
    [阶段五] 纯净的 Yelp 细粒度特征抽取器接口
    """

    def __init__(self, domain_config_name="yelp_restaurant", model_name="model_final"):
        print("正在初始化 Yelp 特征抽取引擎...")
        initialize(config_path="conf")
        # 加载我们阶段二配置的 YAML
        self.cfg = compose("config.yaml", overrides=[f'domain={domain_config_name}', 'model=SBASC'])

        # 实例化结构并挂载我们阶段四强制保存的权重
        self.trainer = Trainer(self.cfg, **self.cfg.domain.params)
        self.trainer.load_model(model_name)
        self.trainer.model.eval()  # 切换为推理模式
        print("引擎加载完成！")

    def extract_sentences_soft(self, sentences: list):
        """
        [软投票改造] 纯净推理接口：输出每个句子在所有维度和情感上的概率分布
        """
        if not sentences:
            return []

        # 1. 调用底层真·张量化推理
        _, _, logits_cat, logits_pol = self.trainer.predict(sentences)

        # 2. 使用 Softmax 将 Logits 转化为 0~1 的概率分布
        probs_cat = F.softmax(logits_cat, dim=1)
        probs_pol = F.softmax(logits_pol, dim=1)

        batch_results = []

        # 3. 将概率张量打包为人类和下游易读的字典格式
        for i in range(len(sentences)):
            sentence_probs = {
                'aspects': {
                    self.trainer.aspect_dict[j]: probs_cat[i][j].item()
                    for j in range(len(self.trainer.aspect_dict))
                },
                'polarities': {
                    self.trainer.polarity_dict[k]: probs_pol[i][k].item()
                    for k in range(len(self.trainer.polarity_dict))
                }
            }
            batch_results.append(sentence_probs)

        return batch_results


# ================= 测试你的纯净接口 =================
if __name__ == "__main__":
    extractor = YelpFeatureExtractor(domain_config_name="yelp-restaurant")

    # 模拟从数据库中取出的 Yelp 单句
    sample_yelp_sentences = [
        "The spicy tuna roll was absolutely amazing.",  # 预期: food, positive
        "We waited 40 minutes for a table, very rude.",  # 预期: service, negative
        "Beautiful patio but the drinks were 20 bucks."  # 预期: ambience/price
    ]

    aspects, polarities = extractor.extract_sentences(sample_yelp_sentences)

    for sent, asp, pol in zip(sample_yelp_sentences, aspects, polarities):
        print(f"文本: {sent}")
        print(f"抽取结果 => 维度: [{asp.upper()}], 情感: [{pol.upper()}]\n")