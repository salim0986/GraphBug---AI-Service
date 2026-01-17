"""
Context Builder Service - Phase 3.5
Combines graph queries, vector search, and code analysis into unified context for LangGraph
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .analyzer import CodeAnalyzer, PRAnalysisRequest, FileChange
from .graph_builder import GraphBuilder
from .vector_builder import VectorBuilder
from .logger import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# MODELS
# ============================================================================

class CodeContext(BaseModel):
    """Context for a single code entity"""
    name: str
    type: str  # function, class, method, etc.
    file: str
    line: int
    code: str
    related_entities: List[Dict[str, Any]] = []
    similar_code: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []


class FileContext(BaseModel):
    """Context for a single file"""
    filename: str
    language: str
    change_type: str  # added, modified, deleted
    additions: int
    deletions: int
    entities: List[CodeContext] = []
    dependencies: List[str] = []
    issues_summary: Dict[str, int] = {}
    complexity_score: int = 0


class PRContext(BaseModel):
    """Unified context for a Pull Request"""
    pr_number: int
    repo_id: str
    title: str
    description: Optional[str] = None
    
    # High-level metrics
    total_files: int
    total_additions: int
    total_deletions: int
    languages: List[str]
    
    # File-level context
    files: List[FileContext]
    
    # Impact analysis
    affected_callers: int = 0
    complexity_hotspots: List[Dict[str, Any]] = []
    high_coupling_files: List[Dict[str, Any]] = []
    
    # Issues aggregation
    critical_issues: List[Dict[str, Any]] = []
    high_issues: List[Dict[str, Any]] = []
    medium_issues: List[Dict[str, Any]] = []
    issues_by_category: Dict[str, int] = {}
    
    # Recommendations
    recommendations: List[str] = []
    requires_deep_review: bool = False
    risk_level: str = "low"  # low, medium, high, critical


# ============================================================================
# CONTEXT BUILDER
# ============================================================================

class ContextBuilder:
    """
    Builds comprehensive context for PR reviews by combining:
    - Code analysis (issues, patterns, smells)
    - Graph queries (dependencies, impact, coupling)
    - Vector search (similar code, duplicates)
    """
    
    def __init__(
        self,
        analyzer: CodeAnalyzer,
        graph_db: GraphBuilder,
        vector_db: VectorBuilder
    ):
        self.analyzer = analyzer
        self.graph_db = graph_db
        self.vector_db = vector_db
    
    async def build_pr_context(
        self,
        pr_number: int,
        repo_id: str,
        title: str,
        description: Optional[str],
        files: List[FileChange],
        base_ref: str = "main",
        head_ref: str = "unknown"
    ) -> PRContext:
        """
        Build comprehensive context for a PR review
        
        This is the main entry point that orchestrates all analysis
        """
        logger.info(f"Building context for PR #{pr_number} in repo {repo_id}")
        
        # 1. Analyze PR with CodeAnalyzer
        pr_analysis = await self.analyzer.analyze_pr(PRAnalysisRequest(
            pr_number=pr_number,
            repo_id=repo_id,
            files=files,
            base_ref=base_ref,
            head_ref=head_ref
        ))
        
        # 2. Build file-level contexts
        file_contexts = []
        for file_result in pr_analysis.file_results:
            file_ctx = await self._build_file_context(
                repo_id=repo_id,
                filename=file_result.filename,
                language=file_result.language,
                file_change=self._find_file_change(files, file_result.filename),
                issues=file_result.issues,
                similar_code=file_result.similar_code,
                related_code=file_result.related_code
            )
            file_contexts.append(file_ctx)
        
        # 3. Extract critical/high issues
        critical_issues = []
        high_issues = []
        medium_issues = []
        
        for file_result in pr_analysis.file_results:
            for issue in file_result.issues:
                issue_dict = {
                    "file": file_result.filename,
                    "severity": issue.severity,
                    "category": issue.category,
                    "title": issue.title,
                    "description": issue.description,
                    "line": issue.line_number,
                    "suggestion": issue.suggestion
                }
                
                if issue.severity == "critical":
                    critical_issues.append(issue_dict)
                elif issue.severity == "high":
                    high_issues.append(issue_dict)
                elif issue.severity == "medium":
                    medium_issues.append(issue_dict)
        
        # 4. Determine languages
        languages = list(set(fc.language for fc in file_contexts if fc.language))
        
        # 5. Generate recommendations
        recommendations = self._generate_recommendations(
            pr_analysis=pr_analysis,
            file_contexts=file_contexts,
            critical_issues=critical_issues,
            high_issues=high_issues
        )
        
        # 6. Calculate risk level
        risk_level = self._calculate_risk_level(
            pr_analysis=pr_analysis,
            critical_issues=critical_issues,
            high_issues=high_issues
        )
        
        # 7. Build PR context
        pr_context = PRContext(
            pr_number=pr_number,
            repo_id=repo_id,
            title=title,
            description=description,
            total_files=len(files),
            total_additions=sum(f.additions for f in files),
            total_deletions=sum(f.deletions for f in files),
            languages=languages,
            files=file_contexts,
            affected_callers=pr_analysis.overall_metrics.get("total_affected_callers", 0),
            complexity_hotspots=pr_analysis.overall_metrics.get("complexity_hotspots", []),
            high_coupling_files=pr_analysis.overall_metrics.get("high_coupling_files", []),
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            issues_by_category=pr_analysis.issues_by_category,
            recommendations=recommendations,
            requires_deep_review=risk_level in ["high", "critical"],
            risk_level=risk_level
        )
        
        logger.info(f"Context built: {len(file_contexts)} files, "
                   f"{len(critical_issues)} critical issues, "
                   f"risk level: {risk_level}")
        
        return pr_context
    
    async def _build_file_context(
        self,
        repo_id: str,
        filename: str,
        language: str,
        file_change: Optional[FileChange],
        issues: List[Any],
        similar_code: List[Any],
        related_code: List[Any]
    ) -> FileContext:
        """Build context for a single file"""
        
        # Get entities in the file from graph
        entities_data = self.graph_db.find_related_by_file(repo_id, filename, limit=50)
        
        # Build entity contexts with similar code from vector search
        entities = []
        for entity_data in entities_data:
            # Map similar code for this entity (from vector search)
            entity_similar = [
                {
                    "file": sim.file,
                    "score": sim.score,
                    "line": getattr(sim, 'line_number', 0),
                    "snippet": getattr(sim, 'code_snippet', '')[:200]
                }
                for sim in similar_code if hasattr(sim, 'file')
            ][:3]  # Top 3 similar
            
            entity_ctx = CodeContext(
                name=entity_data["name"],
                type=entity_data["type"],
                file=filename,
                line=entity_data["line"],
                code=entity_data.get("code", ""),
                related_entities=related_code[:5] if related_code else [],
                similar_code=entity_similar,
                issues=[]
            )
            entities.append(entity_ctx)
        
        # Get file dependencies
        dependencies = self.graph_db.find_file_dependencies(repo_id, filename)
        dep_files = [dep["file"] for dep in dependencies]
        
        # Aggregate issues by severity
        issues_summary = {
            "critical": len([i for i in issues if i.severity == "critical"]),
            "high": len([i for i in issues if i.severity == "high"]),
            "medium": len([i for i in issues if i.severity == "medium"]),
            "low": len([i for i in issues if i.severity == "low"]),
        }
        
        # Calculate complexity score (based on file size and issues)
        complexity_score = 0
        if file_change:
            complexity_score += file_change.additions + file_change.deletions
        complexity_score += sum(issues_summary.values()) * 10
        
        return FileContext(
            filename=filename,
            language=language,
            change_type=file_change.status if file_change else "unknown",
            additions=file_change.additions if file_change else 0,
            deletions=file_change.deletions if file_change else 0,
            entities=entities,
            dependencies=dep_files,
            issues_summary=issues_summary,
            complexity_score=min(complexity_score, 100)  # Cap at 100
        )
    
    def _find_file_change(self, files: List[FileChange], filename: str) -> Optional[FileChange]:
        """Find FileChange object for a filename"""
        for f in files:
            if f.filename == filename:
                return f
        return None
    
    def _generate_recommendations(
        self,
        pr_analysis: Any,
        file_contexts: List[FileContext],
        critical_issues: List[Dict],
        high_issues: List[Dict]
    ) -> List[str]:
        """Generate actionable recommendations for the PR"""
        recommendations = []
        
        # Security recommendations
        security_critical = len([i for i in critical_issues if i["category"] == "security"])
        security_high = len([i for i in high_issues if i["category"] == "security"])
        
        if security_critical > 0:
            recommendations.append(
                f"🚨 CRITICAL: {security_critical} critical security vulnerabilities detected. "
                "Address these immediately before merging."
            )
        
        if security_high > 0:
            recommendations.append(
                f"⚠️ {security_high} high-severity security issues found. "
                "Review and fix before deployment."
            )
        
        # Complexity recommendations
        high_complexity_files = [fc for fc in file_contexts if fc.complexity_score > 70]
        if high_complexity_files:
            files_str = ", ".join([f.filename for f in high_complexity_files[:3]])
            recommendations.append(
                f"📊 High complexity detected in {len(high_complexity_files)} files ({files_str}). "
                "Consider breaking down into smaller changes."
            )
        
        # Coupling recommendations
        if pr_analysis.overall_metrics.get("high_coupling_files"):
            recommendations.append(
                "🔗 High coupling detected between files. "
                "Review dependencies and consider reducing coupling."
            )
        
        # Impact recommendations
        affected_callers = pr_analysis.overall_metrics.get("total_affected_callers", 0)
        if affected_callers > 10:
            recommendations.append(
                f"🎯 This PR affects {affected_callers} calling functions across the codebase. "
                "Ensure comprehensive testing of impacted areas."
            )
        
        # Code quality recommendations
        quality_issues = sum(1 for i in critical_issues + high_issues if i["category"] == "code_quality")
        if quality_issues > 5:
            recommendations.append(
                f"✨ {quality_issues} code quality issues found. "
                "Address these to improve maintainability."
            )
        
        # Default recommendation if no issues
        if not recommendations:
            recommendations.append(
                "✅ No major issues detected. Code looks good for review!"
            )
        
        return recommendations
    
    def _calculate_risk_level(
        self,
        pr_analysis: Any,
        critical_issues: List[Dict],
        high_issues: List[Dict]
    ) -> str:
        """Calculate overall risk level for the PR"""
        
        # Critical if any critical security issues
        if any(i["category"] == "security" for i in critical_issues):
            return "critical"
        
        # Critical if many critical issues
        if len(critical_issues) >= 3:
            return "critical"
        
        # High if critical issues or many high issues
        if len(critical_issues) > 0 or len(high_issues) >= 5:
            return "high"
        
        # High if high impact
        affected_callers = pr_analysis.overall_metrics.get("total_affected_callers", 0)
        if affected_callers > 20:
            return "high"
        
        # Medium if some high issues
        if len(high_issues) > 0:
            return "medium"
        
        # Medium if large changeset
        if pr_analysis.overall_metrics.get("total_additions", 0) > 500:
            return "medium"
        
        return "low"
    
    def get_context_summary(self, pr_context: PRContext) -> str:
        """Generate a human-readable summary of the PR context"""
        summary_lines = [
            f"PR #{pr_context.pr_number}: {pr_context.title}",
            f"Risk Level: {pr_context.risk_level.upper()}",
            "",
            f"📊 Changes: {pr_context.total_additions} additions, {pr_context.total_deletions} deletions across {pr_context.total_files} files",
            f"💻 Languages: {', '.join(pr_context.languages)}",
            "",
            f"🐛 Issues: {len(pr_context.critical_issues)} critical, {len(pr_context.high_issues)} high, {len(pr_context.medium_issues)} medium",
            f"🎯 Impact: {pr_context.affected_callers} affected callers",
            ""
        ]
        
        if pr_context.recommendations:
            summary_lines.append("📝 Recommendations:")
            for rec in pr_context.recommendations:
                summary_lines.append(f"  • {rec}")
        
        return "\n".join(summary_lines)
