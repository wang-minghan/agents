"""
模型能力探测系统

用于评估LLM在不同维度的能力边界，生成能力档案用于智能任务分配。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from langchain_core.language_models import BaseChatModel
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class CapabilityProfile:
    """模型能力档案"""
    
    model_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    optimal_temp: float = 0.7
    response_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_score(self, category: str) -> float:
        """获取指定类别的得分"""
        return self.scores.get(category, 0.0)
    
    def is_strong_in(self, category: str, threshold: float = 0.7) -> bool:
        """判断是否在某个领域具有优势"""
        return self.get_score(category) >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "model_id": self.model_id,
            "scores": self.scores,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "optimal_temp": self.optimal_temp,
            "response_time": self.response_time,
            "metadata": self.metadata,
        }


class CapabilityDetector:
    """
    模型智力边界探测器
    
    评估维度：
    1. 逻辑推理 (logic)
    2. 创意生成 (creativity)
    3. 代码能力 (code)
    4. 数据分析 (analysis)
    5. 沟通能力 (communication)
    """
    
    # 测试集：简化版，后期可扩展为完整benchmark
    TEST_SUITE = {
        "logic": [
            {
                "question": "如果所有A都是B，所有B都是C，那么所有A都是C吗？",
                "expected_keywords": ["是", "正确", "对", "演绎", "三段论"],
                "weight": 1.0,
            },
            {
                "question": "一个房间里有3个开关，分别控制另一个房间的3个灯泡。你只能进入灯泡房间一次，如何确定哪个开关控制哪个灯泡？",
                "expected_keywords": ["打开", "关闭", "热", "温度", "时间"],
                "weight": 1.0,
            },
        ],
        "creativity": [
            {
                "question": "用'时间'、'镜子'、'回声'三个词写一个有趣的故事开头（50字内）",
                "expected_keywords": [],  # 创意题无固定答案，主要看多样性
                "weight": 1.0,
            },
            {
                "question": "为一家AI创业公司设计一个独特的logo概念",
                "expected_keywords": ["神经网络", "未来", "智能", "创新"],
                "weight": 1.0,
            },
        ],
        "code": [
            {
                "question": "写一个Python函数判断字符串是否为回文",
                "expected_keywords": ["def", "return", "==", "[::-1]", "reverse"],
                "weight": 1.0,
            },
            {
                "question": "解释Python中的装饰器是什么，并举例说明",
                "expected_keywords": ["@", "wrapper", "函数", "闭包", "语法糖"],
                "weight": 1.0,
            },
        ],
        "analysis": [
            {
                "question": "给定数据：[10, 20, 15, 30, 25]，计算平均值、中位数、方差",
                "expected_keywords": ["20", "平均", "中位数", "方差", "标准差"],
                "weight": 1.0,
            },
            {
                "question": "如果一家公司的收入连续3年分别增长10%、20%、-5%，总体增长率是多少？",
                "expected_keywords": ["23", "复合", "增长率", "累积"],
                "weight": 1.0,
            },
        ],
        "communication": [
            {
                "question": "向非技术人员解释什么是'云计算'",
                "expected_keywords": ["互联网", "服务器", "存储", "租用", "简单"],
                "weight": 1.0,
            },
            {
                "question": "用一句话总结机器学习的核心思想",
                "expected_keywords": ["数据", "学习", "模式", "预测", "经验"],
                "weight": 1.0,
            },
        ],
    }
    
    # 能力阈值配置
    STRENGTH_THRESHOLD = 0.7
    WEAKNESS_THRESHOLD = 0.4
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化探测器
        
        Args:
            config: 配置字典，可包含自定义测试集、阈值等
        """
        self.config = config or {}
        self.test_suite = self.config.get("test_suite", self.TEST_SUITE)
        self.strength_threshold = self.config.get("strength_threshold", self.STRENGTH_THRESHOLD)
        self.weakness_threshold = self.config.get("weakness_threshold", self.WEAKNESS_THRESHOLD)
    
    def detect(self, model: BaseChatModel, model_id: str = "unknown") -> CapabilityProfile:
        """
        执行完整能力探测
        
        Args:
            model: 待测试的LLM模型
            model_id: 模型标识符
            
        Returns:
            CapabilityProfile: 能力档案
        """
        print(f"\n🔍 开始探测模型能力: {model_id}")
        
        scores = {}
        total_time = 0.0
        
        for category, tests in self.test_suite.items():
            print(f"  📊 测试类别: {category}")
            category_score, avg_time = self._run_tests(model, tests)
            scores[category] = category_score
            total_time += avg_time
            print(f"    └─ 得分: {category_score:.2f}, 平均响应时间: {avg_time:.2f}s")
        
        # 识别优势和劣势
        strengths = self._identify_strengths(scores)
        weaknesses = self._identify_weaknesses(scores)
        
        # 寻找最优temperature（简化版：基于当前配置）
        optimal_temp = self._find_optimal_temp(model, scores)
        
        # 计算平均响应时间
        avg_response_time = total_time / len(self.test_suite) if self.test_suite else 0.0
        
        profile = CapabilityProfile(
            model_id=model_id,
            scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            optimal_temp=optimal_temp,
            response_time=avg_response_time,
            metadata={"test_count": sum(len(tests) for tests in self.test_suite.values())}
        )
        
        print(f"\n✅ 探测完成!")
        print(f"  优势领域: {', '.join(strengths) if strengths else '无显著优势'}")
        print(f"  劣势领域: {', '.join(weaknesses) if weaknesses else '无显著劣势'}")
        print(f"  推荐temperature: {optimal_temp}")
        
        return profile
    
    def _run_tests(self, model: BaseChatModel, tests: List[Dict[str, Any]]) -> Tuple[float, float]:
        """
        运行一组测试
        
        Args:
            model: LLM模型
            tests: 测试列表
            
        Returns:
            (平均得分, 平均响应时间)
        """
        total_score = 0.0
        total_time = 0.0
        
        for idx, test in enumerate(tests, 1):
            start_time = time.time()
            
            try:
                # 调用模型
                response = model.invoke(test["question"])
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # 计算得分
                score = self._score_response(response_text, test)
                total_score += score * test.get("weight", 1.0)
                
                logger.debug(f"测试 {idx}/{len(tests)} 完成，得分: {score:.2f}")
                
            except Exception as e:
                logger.warning(f"测试 {idx}/{len(tests)} 失败: {str(e)}")
                print(f"    ⚠️ 测试失败: {str(e)}")
                total_score += 0.0
            
            elapsed = time.time() - start_time
            total_time += elapsed
        
        avg_score = total_score / len(tests) if tests else 0.0
        avg_time = total_time / len(tests) if tests else 0.0
        
        return avg_score, avg_time
    
    def _score_response(self, response: str, test: Dict[str, Any]) -> float:
        """
        评估回答质量
        
        简化版：基于关键词匹配。后期可扩展为LLM-as-judge或规则引擎
        
        Args:
            response: 模型回答
            test: 测试配置
            
        Returns:
            得分 (0.0-1.0)
        """
        keywords = test.get("expected_keywords", [])
        
        if not keywords:
            # 创意题等无关键词的，简单基于长度和结构判断
            if len(response) > 20 and len(response) < 500:
                return 0.7  # 基础分
            return 0.5
        
        # 关键词匹配
        response_lower = response.lower()
        matched = sum(1 for kw in keywords if kw.lower() in response_lower)
        
        # 计算得分：匹配率 + 长度奖励
        match_ratio = matched / len(keywords) if keywords else 0.0
        length_bonus = 0.1 if 50 < len(response) < 1000 else 0.0
        
        return min(match_ratio + length_bonus, 1.0)
    
    def _identify_strengths(self, scores: Dict[str, float]) -> List[str]:
        """识别优势领域"""
        return [
            category for category, score in scores.items()
            if score >= self.strength_threshold
        ]
    
    def _identify_weaknesses(self, scores: Dict[str, float]) -> List[str]:
        """识别劣势领域"""
        return [
            category for category, score in scores.items()
            if score < self.weakness_threshold
        ]
    
    def _find_optimal_temp(self, model: BaseChatModel, scores: Dict[str, float]) -> float:
        """
        寻找最优temperature
        
        简化版：基于能力档案推荐
        - 逻辑/代码能力强 -> 低temperature (0.3)
        - 创意能力强 -> 高temperature (0.8)
        - 均衡 -> 中等temperature (0.5)
        
        Args:
            model: LLM模型
            scores: 能力得分
            
        Returns:
            推荐的temperature值
        """
        logic_score = scores.get("logic", 0.0)
        code_score = scores.get("code", 0.0)
        creativity_score = scores.get("creativity", 0.0)
        
        # 加权计算
        analytical_weight = (logic_score + code_score) / 2
        creative_weight = creativity_score
        
        if analytical_weight > creative_weight + 0.2:
            return 0.3  # 偏分析型任务
        elif creative_weight > analytical_weight + 0.2:
            return 0.8  # 偏创意型任务
        else:
            return 0.5  # 均衡型
    
    def quick_detect(self, model_id: str, model_config: Dict[str, Any]) -> CapabilityProfile:
        """
        快速探测（基于硬编码规则，无需实际调用模型）
        
        适用场景：
        - 已知模型特性
        - 需要快速初始化
        - 节省API成本
        
        Args:
            model_id: 模型标识
            model_config: 模型配置
            
        Returns:
            CapabilityProfile: 基于规则的能力档案
        """
        logger.info(f"快速探测模式: {model_id}")
        print(f"\n⚡ 快速探测模式: {model_id}")
        
        # 扩展的模型能力档案库
        profiles = {
            "gpt-4o": {
                "scores": {"logic": 0.92, "creativity": 0.88, "code": 0.93, "analysis": 0.90, "communication": 0.92},
                "optimal_temp": 0.7,
            },
            "gpt-4": {
                "scores": {"logic": 0.9, "creativity": 0.85, "code": 0.9, "analysis": 0.85, "communication": 0.9},
                "optimal_temp": 0.7,
            },
            "gpt-3.5-turbo": {
                "scores": {"logic": 0.75, "creativity": 0.7, "code": 0.75, "analysis": 0.7, "communication": 0.8},
                "optimal_temp": 0.7,
            },
            "claude-3-opus": {
                "scores": {"logic": 0.90, "creativity": 0.93, "code": 0.88, "analysis": 0.85, "communication": 0.95},
                "optimal_temp": 0.7,
            },
            "claude-3-sonnet": {
                "scores": {"logic": 0.85, "creativity": 0.88, "code": 0.83, "analysis": 0.80, "communication": 0.90},
                "optimal_temp": 0.7,
            },
            "claude-3": {
                "scores": {"logic": 0.85, "creativity": 0.9, "code": 0.85, "analysis": 0.8, "communication": 0.95},
                "optimal_temp": 0.7,
            },
            "gemini-pro": {
                "scores": {"logic": 0.82, "creativity": 0.80, "code": 0.81, "analysis": 0.83, "communication": 0.85},
                "optimal_temp": 0.7,
            },
            "deepseek-chat": {
                "scores": {"logic": 0.84, "creativity": 0.78, "code": 0.85, "analysis": 0.83, "communication": 0.82},
                "optimal_temp": 0.7,
            },
            "deepseek-reasoner": {
                "scores": {"logic": 0.90, "creativity": 0.72, "code": 0.88, "analysis": 0.91, "communication": 0.80},
                "optimal_temp": 0.6,
            },
            "grok-4.1-fast": {
                "scores": {"logic": 0.86, "creativity": 0.85, "code": 0.86, "analysis": 0.84, "communication": 0.86},
                "optimal_temp": 0.7,
            },
        }
        
        # 模糊匹配（优先匹配更具体的模型名）
        matched_profile = None
        matched_key = None
        best_match_length = 0
        
        for key, profile_data in profiles.items():
            if key in model_id.lower():
                # 选择最长匹配（更具体）
                if len(key) > best_match_length:
                    matched_profile = profile_data
                    matched_key = key
                    best_match_length = len(key)
        
        # 默认配置
        if not matched_profile:
            logger.warning(f"未识别模型 '{model_id}'，使用默认配置")
            matched_profile = {
                "scores": {"logic": 0.6, "creativity": 0.6, "code": 0.6, "analysis": 0.6, "communication": 0.6},
                "optimal_temp": 0.7,
            }
            print(f"  ⚠️ 未识别模型，使用默认配置")
        else:
            logger.info(f"匹配到模型档案: {matched_key}")
        
        scores = matched_profile["scores"]
        strengths = self._identify_strengths(scores)
        weaknesses = self._identify_weaknesses(scores)
        
        profile = CapabilityProfile(
            model_id=model_id,
            scores=scores,
            strengths=strengths,
            weaknesses=weaknesses,
            optimal_temp=matched_profile["optimal_temp"],
            response_time=0.0,
            metadata={
                "mode": "quick_detect", 
                "config": model_config,
                "matched_template": matched_key if matched_key else "default"
            }
        )
        
        logger.info(f"快速探测完成 - 优势: {strengths}, 劣势: {weaknesses}")
        print(f"  ✅ 快速探测完成 (优势: {', '.join(strengths) if strengths else '无'})")
        return profile
