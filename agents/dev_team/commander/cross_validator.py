"""
交叉核查框架

实现Agent互检、结果校验和一致性分析。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
import json
from difflib import SequenceMatcher


@dataclass
class Review:
    """审查意见"""
    reviewer: str
    target: str
    score: float  # 0.0-1.0
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 0.7
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Conflict:
    """冲突信息"""
    agents: List[str]
    point: str
    descriptions: List[str]
    severity: str = "medium"  # low/medium/high
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """校验报告"""
    status: str  # consistent/conflict_found/conflict_resolved
    reviews: List[Review] = field(default_factory=list)
    conflicts: List[Conflict] = field(default_factory=list)
    consistency_score: float = 0.0
    resolution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "status": self.status,
            "reviews_count": len(self.reviews),
            "conflicts_count": len(self.conflicts),
            "consistency_score": self.consistency_score,
            "resolution": self.resolution,
            "metadata": self.metadata,
        }


class CrossValidator:
    """
    Agent互检与结果校验器
    
    功能：
    1. 匿名交叉审查
    2. 差异识别
    3. 冲突解决
    4. 一致性评分
    """
    
    # 配置参数
    CONSISTENCY_THRESHOLD = 0.8
    CONFLICT_SEVERITY_THRESHOLDS = {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.9
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化交叉校验器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.consistency_threshold = self.config.get(
            "consistency_threshold", 
            self.CONSISTENCY_THRESHOLD
        )
        self.error_knowledge_base: Set[str] = set()  # 全局错误知识库
    
    def validate(
        self,
        results: Dict[str, Any],
        agents: List[Any],
        anonymous: bool = True
    ) -> ValidationReport:
        """
        交叉核查流程
        
        Args:
            results: Agent输出结果字典 {agent_name: output}
            agents: Agent列表
            anonymous: 是否匿名审查
            
        Returns:
            ValidationReport: 校验报告
        """
        print(f"\n🔍 启动交叉核查 (匿名模式: {anonymous})...")
        print(f"  待校验结果数: {len(results)}")
        
        # 1. 收集所有Agent的审查意见
        reviews = self._collect_reviews(results, agents, anonymous)
        
        # 2. 分析差异和冲突
        conflicts = self._identify_conflicts(reviews, results)
        
        # 3. 计算一致性得分
        consistency_score = self._calculate_consistency(reviews, conflicts)
        
        # 4. 处理冲突
        if conflicts:
            print(f"  ⚠️ 发现 {len(conflicts)} 个冲突")
            resolution = self._resolve_conflicts(conflicts, results, agents)
            status = "conflict_resolved"
        else:
            print(f"  ✅ 所有结果一致")
            resolution = None
            status = "consistent"
        
        # 5. 更新错误知识库
        self._update_error_knowledge(conflicts)
        
        report = ValidationReport(
            status=status,
            reviews=reviews,
            conflicts=conflicts,
            consistency_score=consistency_score,
            resolution=resolution,
            metadata={
                "anonymous": anonymous,
                "agents_count": len(agents),
                "error_kb_size": len(self.error_knowledge_base)
            }
        )
        
        print(f"  📊 一致性得分: {consistency_score:.2f}")
        return report
    
    def _collect_reviews(
        self,
        results: Dict[str, Any],
        agents: List[Any],
        anonymous: bool
    ) -> List[Review]:
        """
        收集审查意见
        
        每个Agent审查其他Agent的输出
        """
        print("  📝 收集审查意见...")
        reviews = []
        
        # 创建匿名映射
        agent_names = list(results.keys())
        if anonymous:
            # 打乱顺序实现匿名
            import random
            review_assignments = agent_names.copy()
            random.shuffle(review_assignments)
        else:
            review_assignments = agent_names
        
        # 每个Agent审查下一个Agent的结果
        for i, reviewer_name in enumerate(agent_names):
            target_idx = (i + 1) % len(agent_names)
            target_name = review_assignments[target_idx]
            
            if target_name == reviewer_name:
                continue  # 跳过自己
            
            target_result = results.get(target_name, "")
            
            # 执行审查（简化版：基于规则）
            review = self._perform_review(
                reviewer_name,
                target_name if not anonymous else f"Anonymous_{target_idx}",
                target_result
            )
            reviews.append(review)
        
        print(f"    └─ 收集到 {len(reviews)} 份审查意见")
        return reviews
    
    def _perform_review(
        self,
        reviewer: str,
        target: str,
        result: Any
    ) -> Review:
        """
        执行单次审查
        
        简化版：基于规则的审查
        实际应调用 LLM 进行深度分析
        """
        result_str = str(result)
        issues = []
        suggestions = []
        score = 1.0
        
        # 检查常见问题
        if len(result_str) < 50:
            issues.append("输出内容过短，可能不完整")
            score -= 0.2
        
        if "error" in result_str.lower() or "错误" in result_str:
            issues.append("输出中包含错误信息")
            score -= 0.3
        
        # 检查是否包含代码块
        if "<file" not in result_str and "def " not in result_str:
            suggestions.append("建议包含具体代码实现")
            score -= 0.1
        
        # 检查错误知识库
        for known_error in self.error_knowledge_base:
            if known_error in result_str:
                issues.append(f"检测到已知错误模式: {known_error[:50]}...")
                score -= 0.2
                break
        
        score = max(0.0, min(score, 1.0))
        
        return Review(
            reviewer=reviewer,
            target=target,
            score=score,
            issues=issues,
            suggestions=suggestions,
            confidence=0.7
        )
    
    def _identify_conflicts(
        self,
        reviews: List[Review],
        results: Dict[str, Any]
    ) -> List[Conflict]:
        """
        识别冲突
        
        基于审查意见和结果差异
        """
        print("  🔎 识别冲突...")
        conflicts = []
        
        # 1. 基于审查分数的冲突
        low_score_reviews = [r for r in reviews if r.score < 0.5]
        if low_score_reviews:
            for review in low_score_reviews:
                conflict = Conflict(
                    agents=[review.reviewer, review.target],
                    point="低质量输出",
                    descriptions=[f"{review.reviewer} 认为 {review.target} 的输出质量不佳 (得分: {review.score:.2f})"],
                    severity=self._determine_severity(review.score),
                    metadata={"issues": review.issues}
                )
                conflicts.append(conflict)
        
        # 2. 基于结果相似度的冲突
        result_items = list(results.items())
        for i in range(len(result_items)):
            for j in range(i + 1, len(result_items)):
                name1, result1 = result_items[i]
                name2, result2 = result_items[j]
                
                similarity = self._calculate_similarity(str(result1), str(result2))
                
                # 如果结果差异过大，可能存在冲突
                if similarity < 0.3:
                    conflict = Conflict(
                        agents=[name1, name2],
                        point="结果差异显著",
                        descriptions=[
                            f"{name1} 和 {name2} 的输出相似度仅为 {similarity:.2f}",
                            "可能存在理解偏差或实现分歧"
                        ],
                        severity="medium",
                        metadata={"similarity": similarity}
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        计算文本相似度
        
        使用SequenceMatcher算法
        """
        return SequenceMatcher(None, text1, text2).ratio()
    
    def _determine_severity(self, score: float) -> str:
        """确定冲突严重程度"""
        if score < 0.3:
            return "high"
        elif score < 0.6:
            return "medium"
        else:
            return "low"
    
    def _calculate_consistency(
        self,
        reviews: List[Review],
        conflicts: List[Conflict]
    ) -> float:
        """
        计算整体一致性得分
        
        综合考虑审查分数和冲突数量
        """
        if not reviews:
            return 0.0
        
        # 平均审查分数
        avg_review_score = sum(r.score for r in reviews) / len(reviews)
        
        # 冲突惩罚
        conflict_penalty = len(conflicts) * 0.1
        
        consistency_score = max(0.0, avg_review_score - conflict_penalty)
        return consistency_score
    
    def _resolve_conflicts(
        self,
        conflicts: List[Conflict],
        results: Dict[str, Any],
        agents: List[Any]
    ) -> str:
        """
        解决冲突
        
        策略：
        1. 要求冲突方自证
        2. 第三方仲裁
        3. 选择最佳方案
        """
        print("  ⚖️ 解决冲突...")
        
        resolutions = []
        
        for conflict in conflicts:
            if conflict.severity == "high":
                # 高优先级冲突：需要详细分析
                resolution = self._resolve_high_severity_conflict(conflict, results)
                resolutions.append(f"高严重度冲突: {resolution}")
            else:
                # 低/中优先级：简单处理
                resolution = f"记录冲突点: {conflict.point}"
                resolutions.append(resolution)
        
        return "\n".join(resolutions)
    
    def _resolve_high_severity_conflict(
        self,
        conflict: Conflict,
        results: Dict[str, Any]
    ) -> str:
        """
        解决高严重度冲突
        
        简化版：选择质量较高的结果
        实际应引入LLM进行深度分析和仲裁
        """
        # 获取冲突双方的结果
        agent1, agent2 = conflict.agents[:2]
        result1 = results.get(agent1, "")
        result2 = results.get(agent2, "")
        
        # 简单启发式：选择更长的（通常更详细）
        if len(str(result1)) > len(str(result2)):
            return f"选择 {agent1} 的方案（内容更详细）"
        else:
            return f"选择 {agent2} 的方案（内容更详细）"
    
    def _update_error_knowledge(self, conflicts: List[Conflict]):
        """
        更新全局错误知识库
        
        记录常见错误模式以供未来参考
        """
        for conflict in conflicts:
            if conflict.severity == "high":
                # 提取错误模式（简化版）
                error_pattern = f"{conflict.point}:{','.join(conflict.agents)}"
                self.error_knowledge_base.add(error_pattern)
    
    def get_error_knowledge(self) -> List[str]:
        """获取错误知识库内容"""
        return list(self.error_knowledge_base)
    
    def add_error_pattern(self, pattern: str):
        """手动添加错误模式到知识库"""
        self.error_knowledge_base.add(pattern)
    
    def quick_validate(
        self,
        results: Dict[str, Any],
        threshold: float = 0.8
    ) -> bool:
        """
        快速校验
        
        仅检查基本一致性，不进行详细审查
        
        Args:
            results: 结果字典
            threshold: 一致性阈值
            
        Returns:
            是否通过校验
        """
        if len(results) < 2:
            return True
        
        # 比较所有结果的相似度
        result_values = list(results.values())
        similarities = []
        
        for i in range(len(result_values)):
            for j in range(i + 1, len(result_values)):
                sim = self._calculate_similarity(
                    str(result_values[i]),
                    str(result_values[j])
                )
                similarities.append(sim)
        
        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
        return avg_similarity >= threshold
