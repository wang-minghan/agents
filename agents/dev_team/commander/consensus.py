"""
共识机制引擎

实现多Agent间的共识达成，支持投票、辩论、裁决等多种策略。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import json


class ConsensusStrategy(Enum):
    """共识策略枚举"""
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_VOTE = "weighted_vote"
    DEBATE = "debate"
    COMMANDER_JUDGE = "commander_judge"


@dataclass
class Proposal:
    """提案数据结构"""
    author: str
    content: str
    confidence: float = 0.5
    reasoning: str = ""
    votes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConsensusResult:
    """共识结果"""
    final_decision: str
    strategy_used: str
    rounds: int
    proposals: List[Proposal]
    confidence: float
    convergence_achieved: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "final_decision": self.final_decision,
            "strategy_used": self.strategy_used,
            "rounds": self.rounds,
            "confidence": self.confidence,
            "convergence_achieved": self.convergence_achieved,
            "proposals_count": len(self.proposals),
            "metadata": self.metadata,
        }


class ConsensusEngine:
    """
    多Agent共识达成引擎
    
    支持多种共识策略:
    1. majority_vote: 多数投票
    2. weighted_vote: 基于能力加权投票
    3. debate: 辩论模式(多轮对话直到一致)
    4. commander_judge: 指挥官最终裁决
    """
    
    # 配置参数
    MAX_DEBATE_ROUNDS = 3
    CONFIDENCE_THRESHOLD = 0.8
    CONVERGENCE_THRESHOLD = 0.9
    TIME_LIMIT_SECONDS = 300  # 5分钟
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化共识引擎
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.max_rounds = self.config.get("max_debate_rounds", self.MAX_DEBATE_ROUNDS)
        self.confidence_threshold = self.config.get("confidence_threshold", self.CONFIDENCE_THRESHOLD)
        self.convergence_threshold = self.config.get("convergence_threshold", self.CONVERGENCE_THRESHOLD)
    
    def reach_consensus(
        self,
        agents: List[Any],
        task: str,
        strategy: str = "majority_vote",
        weights: Optional[Dict[str, float]] = None,
        commander_judge: Optional[Callable] = None,
    ) -> ConsensusResult:
        """
        达成共识
        
        Args:
            agents: Agent列表
            task: 任务描述
            strategy: 共识策略
            weights: Agent权重字典 (用于weighted_vote)
            commander_judge: 指挥官裁决函数 (用于commander_judge)
            
        Returns:
            ConsensusResult: 共识结果
        """
        print(f"\n🤝 启动共识机制: {strategy}")
        print(f"  参与Agent数: {len(agents)}")
        
        if strategy == ConsensusStrategy.DEBATE.value:
            return self._debate_mode(agents, task)
        elif strategy == ConsensusStrategy.WEIGHTED_VOTE.value:
            return self._weighted_vote(agents, task, weights or {})
        elif strategy == ConsensusStrategy.COMMANDER_JUDGE.value:
            return self._commander_judge(agents, task, commander_judge)
        else:
            return self._majority_vote(agents, task)
    
    def _majority_vote(self, agents: List[Any], task: str) -> ConsensusResult:
        """
        多数投票模式
        
        每个Agent提出方案，统计投票，多数获胜
        """
        print("  📊 执行多数投票...")
        
        # 1. 收集所有提案
        proposals = []
        for agent in agents:
            try:
                # 模拟Agent提案（实际应调用agent.propose(task)）
                proposal = Proposal(
                    author=agent.role_name if hasattr(agent, 'role_name') else str(agent),
                    content=f"Proposal from {agent}",
                    confidence=0.7,
                    reasoning="Initial proposal"
                )
                proposals.append(proposal)
            except Exception as e:
                print(f"    ⚠️ Agent {agent} 提案失败: {e}")
        
        # 2. 投票阶段（简化版：假设每个Agent投票给自己）
        vote_counts = {}
        for proposal in proposals:
            vote_counts[proposal.author] = 1
        
        # 3. 确定获胜者
        if not vote_counts:
            return ConsensusResult(
                final_decision="No consensus reached",
                strategy_used="majority_vote",
                rounds=1,
                proposals=proposals,
                confidence=0.0,
                convergence_achieved=False,
            )
        
        winner = max(vote_counts.items(), key=lambda x: x[1])
        winning_proposal = next(p for p in proposals if p.author == winner[0])
        
        print(f"  ✅ 投票完成，获胜者: {winner[0]} ({winner[1]}票)")
        
        return ConsensusResult(
            final_decision=winning_proposal.content,
            strategy_used="majority_vote",
            rounds=1,
            proposals=proposals,
            confidence=winning_proposal.confidence,
            convergence_achieved=True,
            metadata={"vote_counts": vote_counts}
        )
    
    def _weighted_vote(
        self,
        agents: List[Any],
        task: str,
        weights: Dict[str, float]
    ) -> ConsensusResult:
        """
        加权投票模式
        
        基于Agent能力权重进行投票
        """
        print("  ⚖️ 执行加权投票...")
        
        proposals = []
        weighted_scores = {}
        
        for agent in agents:
            agent_name = agent.role_name if hasattr(agent, 'role_name') else str(agent)
            weight = weights.get(agent_name, 1.0)
            
            proposal = Proposal(
                author=agent_name,
                content=f"Weighted proposal from {agent_name}",
                confidence=0.7 * weight,
                reasoning=f"Weight: {weight}"
            )
            proposals.append(proposal)
            weighted_scores[agent_name] = weight * proposal.confidence
        
        # 选择加权得分最高的
        if not weighted_scores:
            return ConsensusResult(
                final_decision="No consensus",
                strategy_used="weighted_vote",
                rounds=1,
                proposals=proposals,
                confidence=0.0,
                convergence_achieved=False,
            )
        
        winner = max(weighted_scores.items(), key=lambda x: x[1])
        winning_proposal = next(p for p in proposals if p.author == winner[0])
        
        print(f"  ✅ 加权投票完成，获胜者: {winner[0]} (得分: {winner[1]:.2f})")
        
        return ConsensusResult(
            final_decision=winning_proposal.content,
            strategy_used="weighted_vote",
            rounds=1,
            proposals=proposals,
            confidence=winning_proposal.confidence,
            convergence_achieved=True,
            metadata={"weighted_scores": weighted_scores}
        )
    
    def _debate_mode(self, agents: List[Any], task: str) -> ConsensusResult:
        """
        辩论模式
        
        多轮对话直到达成一致或超过最大轮次
        """
        print(f"  🗣️ 启动辩论模式 (最多{self.max_rounds}轮)...")
        
        proposals = []
        
        # 初始提案
        for agent in agents:
            agent_name = agent.role_name if hasattr(agent, 'role_name') else str(agent)
            proposal = Proposal(
                author=agent_name,
                content=f"Initial proposal from {agent_name}",
                confidence=0.6,
                reasoning="Round 0 initial"
            )
            proposals.append(proposal)
        
        # 多轮辩论
        for round_num in range(1, self.max_rounds + 1):
            print(f"    🔄 辩论第 {round_num} 轮...")
            
            # 每个Agent审查其他提案并提出修改意见
            critiques = []
            for agent in agents:
                agent_name = agent.role_name if hasattr(agent, 'role_name') else str(agent)
                other_proposals = [p for p in proposals if p.author != agent_name]
                
                # 模拟批评（实际应调用agent.critique(other_proposals)）
                critique = {
                    "critic": agent_name,
                    "targets": [p.author for p in other_proposals],
                    "feedback": f"Critique from {agent_name} in round {round_num}"
                }
                critiques.append(critique)
            
            # 基于反馈更新提案
            new_proposals = []
            for agent, old_proposal in zip(agents, proposals):
                agent_name = agent.role_name if hasattr(agent, 'role_name') else str(agent)
                relevant_critiques = [c for c in critiques if agent_name in c.get("targets", [])]
                
                # 模拟修订（实际应调用agent.revise(proposal, critiques)）
                updated_proposal = Proposal(
                    author=agent_name,
                    content=f"Revised proposal from {agent_name} (Round {round_num})",
                    confidence=min(old_proposal.confidence + 0.1, 1.0),
                    reasoning=f"Revised based on {len(relevant_critiques)} critiques"
                )
                new_proposals.append(updated_proposal)
            
            # 检查收敛
            if self._check_convergence(proposals, new_proposals):
                print(f"    ✅ 辩论收敛于第 {round_num} 轮")
                proposals = new_proposals
                break
            
            proposals = new_proposals
            
            # 高置信度提前退出
            max_confidence = max(p.confidence for p in proposals)
            if max_confidence >= self.confidence_threshold:
                print(f"    ✅ 达到置信度阈值 ({max_confidence:.2f})")
                break
        
        # 合并提案
        final_decision = self._merge_proposals(proposals)
        avg_confidence = sum(p.confidence for p in proposals) / len(proposals) if proposals else 0.0
        
        return ConsensusResult(
            final_decision=final_decision,
            strategy_used="debate",
            rounds=round_num,
            proposals=proposals,
            confidence=avg_confidence,
            convergence_achieved=round_num < self.max_rounds,
            metadata={"critiques_count": len(critiques)}
        )
    
    def _commander_judge(
        self,
        agents: List[Any],
        task: str,
        judge_func: Optional[Callable]
    ) -> ConsensusResult:
        """
        指挥官裁决模式
        
        收集所有提案后由指挥官做最终决策
        """
        print("  👨‍⚖️ 指挥官裁决模式...")
        
        # 收集提案
        proposals = []
        for agent in agents:
            agent_name = agent.role_name if hasattr(agent, 'role_name') else str(agent)
            proposal = Proposal(
                author=agent_name,
                content=f"Proposal from {agent_name} for judgment",
                confidence=0.7,
                reasoning="Awaiting commander judgment"
            )
            proposals.append(proposal)
        
        # 指挥官裁决
        if judge_func:
            try:
                final_decision = judge_func(proposals, task)
                confidence = 0.9
                print("  ✅ 指挥官已做出裁决")
            except Exception as e:
                print(f"  ⚠️ 指挥官裁决失败: {e}")
                final_decision = "Commander judgment failed"
                confidence = 0.3
        else:
            # 默认：选择第一个提案
            final_decision = proposals[0].content if proposals else "No proposals"
            confidence = 0.5
            print("  ⚠️ 无指挥官函数，使用默认选择")
        
        return ConsensusResult(
            final_decision=final_decision,
            strategy_used="commander_judge",
            rounds=1,
            proposals=proposals,
            confidence=confidence,
            convergence_achieved=True,
            metadata={"judge_provided": judge_func is not None}
        )
    
    def _check_convergence(
        self,
        old_proposals: List[Proposal],
        new_proposals: List[Proposal]
    ) -> bool:
        """
        检查提案是否收敛
        
        简化版：比较内容相似度
        """
        if len(old_proposals) != len(new_proposals):
            return False
        
        # 简单启发式：如果所有置信度都很高且变化很小
        all_high_confidence = all(p.confidence >= self.convergence_threshold for p in new_proposals)
        
        confidence_changes = [
            abs(new.confidence - old.confidence)
            for old, new in zip(old_proposals, new_proposals)
        ]
        small_changes = all(change < 0.1 for change in confidence_changes)
        
        return all_high_confidence and small_changes
    
    def _merge_proposals(self, proposals: List[Proposal]) -> str:
        """
        合并多个提案为最终决策
        
        简化版：选择置信度最高的
        """
        if not proposals:
            return "No proposals to merge"
        
        best_proposal = max(proposals, key=lambda p: p.confidence)
        
        # 可以在这里添加更复杂的合并逻辑
        # 例如：综合多个高置信度提案的优点
        
        return best_proposal.content
    
    def auto_select_strategy(
        self,
        agents: List[Any],
        task_complexity: str = "medium"
    ) -> str:
        """
        根据任务复杂度自动选择共识策略
        
        Args:
            agents: Agent列表
            task_complexity: 任务复杂度 (simple/medium/complex)
            
        Returns:
            推荐的策略名称
        """
        agent_count = len(agents)
        
        if task_complexity == "simple" or agent_count <= 2:
            return ConsensusStrategy.MAJORITY_VOTE.value
        elif task_complexity == "complex" or agent_count >= 5:
            return ConsensusStrategy.DEBATE.value
        else:
            return ConsensusStrategy.WEIGHTED_VOTE.value
