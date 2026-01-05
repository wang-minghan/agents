"""
AI Commander - 智能任务指挥官

整合能力探测、共识机制、交叉核查等功能的高级协作编排器。
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents.dev_team.commander.base_orchestrator import BaseOrchestrator, default_agent_factory
from agents.dev_team.commander.capability_detector import CapabilityDetector, CapabilityProfile
from agents.dev_team.commander.consensus import ConsensusEngine, ConsensusResult
from agents.dev_team.commander.cross_validator import CrossValidator, ValidationReport
from agents.dev_team.interfaces import Agent

# 配置日志
logger = logging.getLogger(__name__)


class Commander(BaseOrchestrator):
    """
    AI Commander - 增强版协作编排器
    
    在基础协作编排之上新增：
    1. 模型能力探测与智能匹配
    2. 多Agent共识机制
    3. 交叉核查与质量保证
    4. 动态任务分配
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        output_dir: str = None,
        code_executor = None,
        agent_factory = None,
        enable_capability_detection: Optional[bool] = None,
        enable_consensus: Optional[bool] = None,
        enable_cross_validation: Optional[bool] = None,
    ):
        """
        初始化Commander
        
        Args:
            config: 配置字典
            output_dir: 输出目录
            code_executor: 代码执行器
            agent_factory: Agent工厂函数
            enable_capability_detection: 启用能力探测
            enable_consensus: 启用共识机制
            enable_cross_validation: 启用交叉核查
        """
        if agent_factory is None:
            agent_factory = default_agent_factory
        super().__init__(config, output_dir, code_executor, agent_factory)
        
        # 初始化Commander模块（延迟初始化，节省资源）
        self._capability_detector = None
        self._consensus_engine = None
        self._cross_validator = None
        self._commander_config = config.get("commander", {})
        self.mode = "auto"
        self.enable_capability_detection = True if enable_capability_detection is None else enable_capability_detection
        self.enable_consensus = False if enable_consensus is None else enable_consensus
        self.enable_cross_validation = False if enable_cross_validation is None else enable_cross_validation
        
        # 能力档案存储
        self.capability_profiles: Dict[str, CapabilityProfile] = {}
        
        # 性能统计
        self._performance_metrics = {
            "capability_detection_time": 0.0,
            "consensus_time": 0.0,
            "validation_time": 0.0,
        }
        
        logger.info("AI Commander 已初始化（自动模式）")
        
        print(f"\n🎖️ AI Commander 已初始化（自动模式）")
    
    @property
    def capability_detector(self) -> CapabilityDetector:
        """延迟初始化能力探测器"""
        if self._capability_detector is None:
            self._capability_detector = CapabilityDetector(
                self._commander_config.get("capability_detection", {})
            )
        return self._capability_detector
    
    @property
    def consensus_engine(self) -> ConsensusEngine:
        """延迟初始化共识引擎"""
        if self._consensus_engine is None:
            self._consensus_engine = ConsensusEngine(
                self._commander_config.get("consensus", {})
            )
        return self._consensus_engine
    
    @property
    def cross_validator(self) -> CrossValidator:
        """延迟初始化交叉验证器"""
        if self._cross_validator is None:
            self._cross_validator = CrossValidator(
                self._commander_config.get("cross_validation", {})
            )
        return self._cross_validator
    
    def initialize_team(self, planner_result: Dict[str, Any]):
        """
        增强版团队初始化
        
        在基础初始化后，添加能力探测和智能匹配
        """
        # 调用父类的初始化
        super().initialize_team(planner_result)

        self._apply_mode()
        
        # 能力探测
        if self.enable_capability_detection:
            self._detect_agent_capabilities()

    def _apply_mode(self) -> None:
        complexity = self._estimate_task_complexity()
        if complexity == "simple":
            self.mode = "local"
            self.enable_consensus = False
            self.enable_cross_validation = False
        else:
            self.mode = "optimal"
            self.enable_consensus = True
            self.enable_cross_validation = True
        print(f"  模式选择: {self.mode} ({complexity})")
    
    def _detect_agent_capabilities(self):
        """探测所有Agent的能力"""
        import time
        start_time = time.time()
        
        print(f"\n🔍 正在探测团队成员能力...")
        logger.info(f"开始能力探测，团队规模: {len(self.agents)}")
        
        # 批量探测所有Agent
        agents_to_detect = list(self.agents)
        if getattr(self, "qa_agents", None):
            agents_to_detect.extend(self.qa_agents)
        
        for agent in agents_to_detect:
            if not hasattr(agent, 'role_name'):
                logger.warning(f"Agent缺少role_name属性，跳过: {agent}")
                continue
                
            try:
                model_id = self._extract_model_id(agent)
                profile = self.capability_detector.quick_detect(
                    model_id=model_id,
                    model_config={}
                )
                self.capability_profiles[agent.role_name] = profile
                print(f"  ✓ {agent.role_name}: {', '.join(profile.strengths)}")
                logger.info(f"探测完成 - {agent.role_name}: 得分 {profile.scores}")
                
            except Exception as e:
                logger.error(f"探测Agent能力时出错 ({agent.role_name}): {str(e)}")
                # 使用默认档案
                self.capability_profiles[agent.role_name] = self._get_default_profile(
                    agent.role_name
                )
        
        elapsed = time.time() - start_time
        self._performance_metrics["capability_detection_time"] = elapsed
        logger.info(f"能力探测完成，耗时: {elapsed:.2f}秒")
    
    def _extract_model_id(self, agent: Agent) -> str:
        """提取Agent的模型ID"""
        if hasattr(agent, 'llm'):
            if hasattr(agent.llm, 'model_name'):
                return agent.llm.model_name
            elif hasattr(agent.llm, 'model'):
                return agent.llm.model
        return 'unknown'
    
    def _get_default_profile(self, agent_name: str) -> CapabilityProfile:
        """获取默认能力档案"""
        return CapabilityProfile(
            model_id=f"{agent_name}_default",
            scores={"logic": 0.6, "creativity": 0.6, "code": 0.6, 
                   "analysis": 0.6, "communication": 0.6},
            strengths=[],
            weaknesses=[],
            optimal_temp=0.7,
            response_time=0.0,
            metadata={"type": "default"}
        )
    
    def run_collaboration(self, max_rounds: int = 5):
        """
        增强版协作流程
        
        在每轮迭代后可选启用共识机制和交叉核查
        """
        started_at = self._utcnow()
        self.run_reports = []
        start_round = self._resume_if_available(max_rounds)
        
        if not self.agents:
            print("❌ 错误: 团队未初始化")
            report = self._build_report("no_engineers", started_at)
            self._write_report(report)
            return {
                "status": "error",
                "error": "no_engineers",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        review_report = self._ensure_review_artifacts()
        self.shared_memory.global_context["review_artifacts"] = review_report
        if review_report.get("status") not in ("passed", "skipped"):
            report = self._build_report("review_missing", started_at)
            report["review"] = review_report
            self._write_report(report)
            return {
                "status": "review_missing",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }
        
        if start_round > max_rounds:
            report = self._build_report("max_rounds_reached", started_at)
            self._write_report(report)
            self._save_resume_state(max_rounds, "max_rounds_reached")
            return {
                "status": "max_rounds_reached",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        ui_design_report = self._prepare_ui_design_assets()
        self.shared_memory.global_context["ui_design_report"] = ui_design_report
        allow_missing_ui = self._allow_missing_ui_baseline()
        if (
            ui_design_report.get("status") == "failed"
            and self.config.get("ui_design", {}).get("required", True)
            and not allow_missing_ui
        ):
            report = self._build_report("ui_design_failed", started_at)
            report["ui_design"] = ui_design_report
            self._write_report(report)
            self._save_resume_state(0, "ui_design_failed")
            return {
                "status": "ui_design_failed",
                "outputs": self.shared_memory.get_all_outputs(),
                "report": report,
            }

        print(f"\n🚀 AI Commander 协作流程启动 (最大轮次: {max_rounds})...")
        
        run_status = "max_rounds_reached"
        testing_cfg = self.config.get("testing", {})
        testing_enabled = testing_cfg.get("enabled", True)
        
        for round_num in range(start_round, max_rounds + 1):
            print(f"\n{'='*60}")
            print(f"🔄 第 {round_num} 轮迭代")
            print(f"{'='*60}")
            
            round_report = {
                "round": round_num,
                "agents": [],
                "tests": {},
                "consensus": None,
                "validation": None,
                "qa_feedback_recorded": False,
            }
            self._round_saved_files = set()
            failure_reasons: List[str] = []
            
            # 1. Agent工作阶段
            print("\n📝 阶段1: Agent工作")
            max_workers = min(4, max(1, len(self.agents)))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(self._run_agent, agent) for agent in self.agents]
                for future in as_completed(futures):
                    round_report["agents"].append(future.result())

            refactor_suggestions = self._collect_refactor_suggestions(self.output_dir)
            round_report["refactor_suggestions"] = refactor_suggestions
            self.shared_memory.global_context["refactor_suggestions"] = refactor_suggestions
            if refactor_suggestions:
                print("  ⚠️ 发现需要拆解的长函数，已生成建议。")
            
            # 2. 共识机制（可选）
            if self.enable_consensus and len(self.agents) > 1:
                print("\n🤝 阶段2: 共识达成")
                consensus_result = self._reach_consensus(round_num)
                round_report["consensus"] = consensus_result.to_dict()
                if self._should_block_on_consensus(consensus_result):
                    failure_reasons.append("consensus_failed")
            
            # 3. 测试执行
            print("\n🧪 阶段3: 自动化测试")
            if testing_enabled:
                test_results = self.code_executor.run_tests(str(self.output_dir))
            else:
                test_results = "SKIPPED: Testing disabled by config."
            self.shared_memory.global_context["latest_test_results"] = test_results
            
            summary = test_results.splitlines()[0] if test_results else "No output"
            test_status = self._classify_test_result(test_results)
            self.shared_memory.global_context["latest_test_status"] = test_status
            
            round_report["tests"] = {"status": test_status, "summary": summary}
            print(f"  测试状态: {test_status}")

            ui_test_result = "SKIPPED: No UI tests."
            if self._requires_ui_baseline() and hasattr(self.code_executor, "run_ui_tests"):
                print("  🧪 [System] 正在执行 UI 测试...")
                ui_test_result = self.code_executor.run_ui_tests(str(self.output_dir))
            ui_test_status = self._classify_test_result(ui_test_result)
            round_report["ui_tests"] = {
                "status": ui_test_status,
                "summary": ui_test_result.splitlines()[0] if ui_test_result else "No output",
            }

            coverage_result = "SKIPPED: No coverage run."
            if hasattr(self.code_executor, "run_coverage"):
                print("  🧪 [System] 正在执行覆盖率统计...")
                coverage_result = self.code_executor.run_coverage(str(self.output_dir))
            coverage_status = self._classify_test_result(coverage_result)
            round_report["coverage"] = {
                "status": coverage_status,
                "summary": coverage_result.splitlines()[0] if coverage_result else "No output",
            }

            input_result = "SKIPPED: No input contract tests."
            if hasattr(self.code_executor, "run_input_contract_tests"):
                print("  🧪 [System] 正在执行输入契约测试...")
                input_result = self.code_executor.run_input_contract_tests(str(self.output_dir))
            input_status = self._classify_test_result(input_result)
            round_report["input_contract"] = {
                "status": input_status,
                "summary": input_result.splitlines()[0] if input_result else "No output",
            }
            
            # 4. 交叉核查（可选）
            if self.enable_cross_validation and len(self.agents) > 1:
                print("\n🔍 阶段4: 交叉核查")
                validation_report = self._cross_validate()
                round_report["validation"] = validation_report.to_dict()
                if self._should_block_on_validation(validation_report):
                    failure_reasons.append("validation_failed")
            
            # 5. QA审查
            if getattr(self, "qa_agents", None):
                print(f"\n👨‍💼 阶段5: QA审查")
                qa_reports = []
                max_workers = min(4, max(1, len(self.qa_agents)))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_qa_agent,
                            agent,
                            round_num,
                            test_status,
                        )
                        for agent in self.qa_agents
                    ]
                for future in as_completed(futures):
                    qa_reports.append(future.result())
                round_report["qa_feedback"] = qa_reports
                round_report["qa_feedback_recorded"] = True
                qa_gate_cfg = self.config.get("quality_gates", {}).get("qa", {})
                if qa_gate_cfg.get("enabled", True) and self._qa_feedback_failed(qa_reports):
                    failure_reasons.append("qa_failed")
            
            # 6. 判断是否结束
            if test_status == "passed":
                print("\n✨ 所有测试通过！")
                if input_status in ("failed", "error"):
                    print("  ❌ 输入契约测试未通过，进入修复回合。")
                    failure_reasons.append("input_contract_failed")
                if self._requires_ui_baseline() and self._should_require_ui_tests():
                    if ui_test_status in ("failed", "error", "skipped", "unknown"):
                        print("  ❌ UI 测试未通过或缺失，进入修复回合。")
                        failure_reasons.append("ui_tests_failed")
                if self._should_require_coverage():
                    if coverage_status in ("failed", "error", "skipped", "unknown"):
                        print("  ❌ 覆盖率统计未通过或缺失，进入修复回合。")
                        failure_reasons.append("coverage_failed")
                if self._requires_ui_baseline():
                    ui_check = self._check_ui_evidence()
                    round_report["ui_evidence"] = ui_check
                    if ui_check["status"] != "passed":
                        print("  ❌ UI 证据不完整，进入修复回合。")
                        failure_reasons.append("ui_evidence_missing")
                sim_result = "SKIPPED: No user simulation."
                if hasattr(self.code_executor, "run_user_simulation"):
                    print("  🧭 [System] 正在执行用户模拟测试...")
                    sim_result = self.code_executor.run_user_simulation(str(self.output_dir))
                sim_status = self._classify_test_result(sim_result)
                round_report["user_simulation"] = {
                    "status": sim_status,
                    "summary": sim_result.splitlines()[0] if sim_result else "No output",
                }
                if self._requires_ui_baseline() and self._should_require_ui_simulation():
                    if sim_status not in ("passed",):
                        print("  ❌ 用户模拟测试缺失或失败，进入修复回合。")
                        failure_reasons.append("user_simulation_failed")
                elif sim_status not in ("passed", "skipped", "unknown"):
                    print("  ❌ 用户模拟测试未通过，进入修复回合。")
                    failure_reasons.append("user_simulation_failed")
                acceptance_criteria = self._get_acceptance_criteria(self._get_requirements_payload())
                acceptance_report = self._verify_acceptance_checklist(acceptance_criteria)
                round_report["acceptance"] = acceptance_report
                if acceptance_report.get("status") == "failed":
                    print("  ❌ 验收清单未完成，进入修复回合。")
                    failure_reasons.append("acceptance_failed")
                if not failure_reasons:
                    self._write_evidence_manifest(round_report)
                    self._archive_evidence()
                    approved = self._run_approval_gate(round_num, test_status, round_report)
                    if approved:
                        run_status = "passed"
                        self.run_reports.append(round_report)
                        break
                    print("  ⚠️ 审批未通过，进入修复回合。")
                    failure_reasons.append("approval_failed")

            else:
                if test_status in ("failed", "error"):
                    failure_reasons.append("tests_failed")
                elif test_status in ("skipped", "unknown") and self._should_require_tests():
                    failure_reasons.append("tests_failed")
                if self._requires_ui_baseline() and self._should_require_ui_tests():
                    if ui_test_status in ("failed", "error", "skipped", "unknown"):
                        failure_reasons.append("ui_tests_failed")
                if self._should_require_coverage():
                    if coverage_status in ("failed", "error", "skipped", "unknown"):
                        failure_reasons.append("coverage_failed")

            if failure_reasons:
                round_report["failure_reasons"] = failure_reasons
                self._write_bug_card(failure_reasons[0], round_report)
                self._write_evidence_manifest(round_report)
                self._archive_evidence()
                self.run_reports.append(round_report)
                self._save_resume_state(round_num, "in_progress")
                self._sync_iteration_artifacts()
                run_status = failure_reasons[0]
                if round_num < max_rounds:
                    continue
                break
            
            self.run_reports.append(round_report)
            self._save_resume_state(round_num, "in_progress")
            self._sync_iteration_artifacts()
        
        # 生成最终报告
        report = self._build_report(run_status, started_at)
        self._write_report(report)
        self._save_resume_state(len(self.run_reports), run_status)
        self._sync_iteration_artifacts()
        
        return {
            "status": run_status,
            "outputs": self.shared_memory.get_all_outputs(),
            "report": report,
            "capability_profiles": {
                name: profile.to_dict() 
                for name, profile in self.capability_profiles.items()
            }
        }
    
    def _run_agent(self, agent: Agent) -> Dict[str, Any]:
        """运行单个Agent"""
        return self._run_agent_once(agent)
    
    def _reach_consensus(self, round_num: int) -> ConsensusResult:
        """达成共识"""
        import time
        start_time = time.time()
        
        try:
            # 获取所有Agent的输出
            all_outputs = self.shared_memory.get_all_outputs()
            
            # 确定共识策略
            task_complexity = self._estimate_task_complexity()
            strategy = self.consensus_engine.auto_select_strategy(
                self.agents,
                task_complexity
            )
            
            logger.info(f"第{round_num}轮共识，策略: {strategy}, 复杂度: {task_complexity}")
            
            # 构建权重（基于能力档案）
            weights = self._calculate_agent_weights()
            
            # 达成共识
            consensus_result = self.consensus_engine.reach_consensus(
                agents=self.agents,
                task=f"Round {round_num} collaboration",
                strategy=strategy,
                weights=weights
            )
            
            elapsed = time.time() - start_time
            self._performance_metrics["consensus_time"] += elapsed
            logger.info(f"共识达成完成，耗时: {elapsed:.2f}秒，置信度: {consensus_result.confidence}")
            
            return consensus_result
            
        except Exception as e:
            logger.error(f"共识达成失败: {str(e)}")
            # 返回默认共识结果
            return ConsensusResult(
                final_decision="继续协作",
                confidence=0.5,
                votes={},
                strategy="fallback",
                rounds=1
            )
    
    def _calculate_agent_weights(self) -> Dict[str, float]:
        """计算Agent权重（基于能力档案）"""
        weights = {}
        for agent in self.agents:
            agent_name = agent.role_name if hasattr(agent, 'role_name') else str(agent)
            if agent_name in self.capability_profiles:
                profile = self.capability_profiles[agent_name]
                # 基于代码和逻辑能力的加权平均
                weights[agent_name] = (
                    profile.get_score("code") * 0.6 + 
                    profile.get_score("logic") * 0.4
                )
            else:
                weights[agent_name] = 1.0
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
    
    def _cross_validate(self) -> ValidationReport:
        """执行交叉核查"""
        import time
        start_time = time.time()
        
        try:
            # 获取所有Agent的输出
            all_outputs = self.shared_memory.get_all_outputs()
            
            # 提取每个Agent的最新输出
            results = self._extract_latest_outputs(all_outputs)
            
            if len(results) < 2:
                logger.warning("输出数量不足，跳过交叉核查")
                return self._create_empty_validation_report()
            
            logger.info(f"开始交叉核查，结果数: {len(results)}")
            
            # 执行交叉核查
            validation_report = self.cross_validator.validate(
                results=results,
                agents=self.agents,
                anonymous=True
            )
            
            elapsed = time.time() - start_time
            self._performance_metrics["validation_time"] += elapsed
            logger.info(f"交叉核查完成，耗时: {elapsed:.2f}秒，冲突数: {len(validation_report.conflicts)}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"交叉核查失败: {str(e)}")
            return self._create_empty_validation_report()
    
    def _extract_latest_outputs(self, all_outputs: Dict[str, List[Any]]) -> Dict[str, Any]:
        """提取所有Agent的最新输出"""
        results = {}
        for agent in self.agents:
            agent_name = agent.role_name if hasattr(agent, 'role_name') else str(agent)
            if agent_name in all_outputs and all_outputs[agent_name]:
                results[agent_name] = all_outputs[agent_name][-1]
        return results
    
    def _create_empty_validation_report(self) -> ValidationReport:
        """创建空的验证报告"""
        from agents.dev_team.commander.cross_validator import ValidationReport
        return ValidationReport(
            reviews={},
            conflicts=[],
            consistency_score=1.0,
            is_valid=True
        )
    
    def _estimate_task_complexity(self) -> str:
        """
        估算任务复杂度
        
        基于需求长度和Agent数量的启发式估算
        """
        requirements = self.shared_memory.global_context.get("requirements", "")
        
        if isinstance(requirements, str):
            req_length = len(requirements)
        else:
            req_length = len(str(requirements))
        
        agent_count = len(self.agents)
        
        if req_length > 1000 or agent_count >= 5:
            return "complex"
        elif req_length > 500 or agent_count >= 3:
            return "medium"
        else:
            return "simple"
    
    def get_capability_summary(self) -> Dict[str, Any]:
        """获取团队能力总结"""
        if not self.capability_profiles:
            return {"message": "能力探测未启用"}
        
        # 计算团队统计信息
        all_scores = {category: [] for category in ["logic", "creativity", "code", "analysis", "communication"]}
        
        for profile in self.capability_profiles.values():
            for category, score in profile.scores.items():
                if category in all_scores:
                    all_scores[category].append(score)
        
        team_avg_scores = {
            category: sum(scores) / len(scores) if scores else 0.0
            for category, scores in all_scores.items()
        }
        
        summary = {
            "team_size": len(self.capability_profiles),
            "team_average_scores": team_avg_scores,
            "team_strengths": [cat for cat, score in team_avg_scores.items() if score >= 0.7],
            "team_weaknesses": [cat for cat, score in team_avg_scores.items() if score < 0.5],
            "profiles": {},
            "performance_metrics": self._performance_metrics
        }
        
        for name, profile in self.capability_profiles.items():
            summary["profiles"][name] = {
                "model": profile.model_id,
                "strengths": profile.strengths,
                "weaknesses": profile.weaknesses,
                "scores": profile.scores,
                "optimal_temp": profile.optimal_temp
            }
        
        return summary
    
    def save_capability_profiles(self, filepath: Optional[Path] = None):
        """保存能力档案到文件"""
        if not filepath:
            filepath = self.output_dir / "capability_profiles.json"
        
        profiles_data = {
            name: profile.to_dict()
            for name, profile in self.capability_profiles.items()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(profiles_data, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 能力档案已保存: {filepath}")
