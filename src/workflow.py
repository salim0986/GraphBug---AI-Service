"""
LangGraph Workflow for AI Code Review - Phase 4.1
Orchestrates the entire review process using LangGraph StateGraph
"""

from typing import TypedDict, List, Dict, Any, Annotated, Optional, Literal
from typing_extensions import NotRequired
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dataclasses import dataclass, field, asdict
from datetime import datetime
from .logger import setup_logger
from .context_builder import ContextBuilder
from .gemini_client import GeminiClient, create_gemini_client
from .analyzer import FileChange
from .temporary_graph import TemporaryGraphBuilder, TemporaryVectorBuilder
from .context_merger import ContextMerger
from .parser import UniversalParser
from sentence_transformers import SentenceTransformer

logger = setup_logger(__name__)


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ReviewState(TypedDict):
    """
    Complete state for the code review workflow
    
    This state is passed through all nodes and maintains the entire
    review context and progress.
    """
    # PR Metadata
    pr_number: int
    repo_id: str
    pr_title: str
    pr_description: Optional[str]
    base_ref: NotRequired[str]  # Base branch (e.g., "main")
    head_ref: NotRequired[str]  # Head branch (e.g., "feature/new-feature")
    
    # Input Data
    files: List[Dict[str, Any]]  # Raw file changes from GitHub
    
    # Context (from Phase 3)
    pr_context: NotRequired[Dict[str, Any]]  # PRContext from context_builder
    
    # Review Strategy
    review_strategy: NotRequired[Literal["quick", "standard", "deep"]]
    requires_deep_review: NotRequired[bool]
    risk_level: NotRequired[str]
    
    # File Prioritization
    high_priority_files: NotRequired[List[str]]
    medium_priority_files: NotRequired[List[str]]
    low_priority_files: NotRequired[List[str]]
    
    # Review Generation
    file_reviews: NotRequired[Dict[str, Dict[str, Any]]]  # filename -> review
    overall_summary: NotRequired[str]
    
    # Model Selection
    selected_model: NotRequired[str]  # gemini-2.5-flash-lite, flash, or pro
    
    # Progress Tracking
    status: NotRequired[str]  # queued, analyzing, reviewing, completed, failed
    current_step: NotRequired[str]
    processed_files: NotRequired[int]
    total_files: NotRequired[int]
    
    # Error Handling
    errors: NotRequired[List[Dict[str, Any]]]
    retry_count: NotRequired[int]
    
    # Timing
    started_at: NotRequired[str]
    completed_at: NotRequired[str]
    
    # Messages (for LangGraph message passing)
    messages: Annotated[List[Dict[str, Any]], add_messages]


# ============================================================================
# WORKFLOW CONFIGURATION
# ============================================================================

@dataclass
class WorkflowConfig:
    """Configuration for the review workflow"""
    
    # Model selection thresholds
    quick_review_max_files: int = 3
    quick_review_max_additions: int = 100
    
    standard_review_max_files: int = 10
    standard_review_max_additions: int = 500
    
    # Gemini model configuration
    flash_lite_model: str = "gemini-2.5-flash-latest"
    flash_model: str = "gemini-2.5-flash-latest"
    pro_model: str = "gemini-2.5-pro-latest"
    
    # Rate limiting
    max_requests_per_minute: int = 60
    max_tokens_per_request: int = 30000
    
    # Retry configuration
    max_retries: int = 3
    retry_delay_seconds: int = 2
    
    # Review configuration
    max_files_per_batch: int = 5
    context_window_tokens: int = 25000
    
    # Priority thresholds
    high_priority_complexity: int = 70
    high_priority_issues: int = 3  # critical + high issues


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

class CodeReviewWorkflow:
    """
    LangGraph workflow for orchestrating AI code reviews
    
    Workflow Steps:
    1. START -> analyze (build context)
    2. analyze -> route (determine review strategy)
    3. route -> review_quick/review_standard/review_deep
    4. review_* -> aggregate
    5. aggregate -> END
    """
    
    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        context_builder: Optional[ContextBuilder] = None,
        gemini_client: Optional[GeminiClient] = None
    ):
        self.config = config or WorkflowConfig()
        self.context_builder = context_builder
        self.gemini_client = gemini_client or create_gemini_client()
        self.graph = self._build_graph()
        self.compiled = None
        
        # Initialize for temporary GraphRAG (lazy loading)
        self._parser = None
        self._embed_model = None
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph StateGraph"""
        
        # Create graph with ReviewState
        workflow = StateGraph(ReviewState)
        
        # Add nodes (we'll implement these in subsequent phases)
        workflow.add_node("analyze", self._analyze_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("review_quick", self._review_quick_node)
        workflow.add_node("review_standard", self._review_standard_node)
        workflow.add_node("review_deep", self._review_deep_node)
        workflow.add_node("aggregate", self._aggregate_node)
        workflow.add_node("handle_error", self._error_handler_node)
        
        # Define edges
        workflow.add_edge(START, "analyze")
        workflow.add_edge("analyze", "route")
        
        # Conditional edges from route node
        workflow.add_conditional_edges(
            "route",
            self._route_decision,
            {
                "quick": "review_quick",
                "standard": "review_standard",
                "deep": "review_deep",
                "error": "handle_error"
            }
        )
        
        # All review nodes go to aggregate
        workflow.add_edge("review_quick", "aggregate")
        workflow.add_edge("review_standard", "aggregate")
        workflow.add_edge("review_deep", "aggregate")
        
        # Aggregate goes to END
        workflow.add_edge("aggregate", END)
        
        # Error handler can retry or end
        workflow.add_conditional_edges(
            "handle_error",
            self._error_decision,
            {
                "retry": "analyze",
                "end": END
            }
        )
        
        return workflow
    
    def compile(self):
        """Compile the workflow for execution"""
        self.compiled = self.graph.compile()
        logger.info("Code review workflow compiled successfully")
        return self.compiled
    
    def _get_parser(self) -> UniversalParser:
        """Lazy load UniversalParser for temporary GraphRAG"""
        if self._parser is None:
            self._parser = UniversalParser()
        return self._parser
    
    def _get_embed_model(self) -> SentenceTransformer:
        """Lazy load embedding model for temporary vectors"""
        if self._embed_model is None:
            logger.info("[TempGraphRAG] Loading sentence-transformers model...")
            self._embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        return self._embed_model
    
    async def _build_temporary_graphrag(
        self,
        pr_files: List[Dict[str, Any]],
        github_client: Optional[Any] = None
    ) -> tuple[Optional[TemporaryGraphBuilder], Optional[TemporaryVectorBuilder]]:
        """
        Build temporary in-memory GraphRAG for PR files
        
        Args:
            pr_files: List of file dicts with filename, status, patch, etc.
            github_client: Optional GitHub client to fetch full file content
            
        Returns:
            Tuple of (temp_graph, temp_vector) or (None, None) on error
        """
        try:
            logger.info(f"[TempGraphRAG] Building temporary GraphRAG for {len(pr_files)} files")
            
            # Initialize builders
            parser = self._get_parser()
            temp_graph = TemporaryGraphBuilder(parser)
            temp_vector = TemporaryVectorBuilder(self._get_embed_model())
            
            # Process each file
            processed_count = 0
            for file_dict in pr_files:
                try:
                    filename = file_dict.get("filename", "")
                    language = file_dict.get("language")
                    patch = file_dict.get("patch", "")
                    status = file_dict.get("status", "modified")
                    
                    # Skip deleted files
                    if status == "deleted":
                        continue
                    
                    # For new/modified files, try to reconstruct full content from patch
                    # If patch is incomplete, we could fetch from GitHub API
                    # For now, use patch as best-effort content
                    
                    # Extract code lines from patch (remove diff markers)
                    code_lines = []
                    for line in patch.split('\n'):
                        if line.startswith('+') and not line.startswith('+++'):
                            code_lines.append(line[1:])  # Remove + prefix
                        elif line.startswith(' '):
                            code_lines.append(line[1:])  # Context line
                    
                    content = '\n'.join(code_lines)
                    
                    if not content.strip():
                        logger.warning(f"[TempGraphRAG] No content for {filename}, skipping")
                        continue
                    
                    # Process file with tree-sitter
                    file_node = temp_graph.process_file(filename, content, language or "text")
                    
                    # Add nodes to vector index
                    if file_node.nodes:
                        temp_vector.add_nodes(file_node.nodes)
                        processed_count += 1
                    
                except Exception as e:
                    logger.error(f"[TempGraphRAG] Error processing {filename}: {e}")
                    continue
            
            # Build file dependencies
            temp_graph.build_dependencies()
            
            logger.info(
                f"[TempGraphRAG] Built temporary GraphRAG: "
                f"{processed_count} files, {len(temp_graph.nodes)} nodes, "
                f"{len(temp_vector.vectors)} vectors"
            )
            
            return temp_graph, temp_vector
            
        except Exception as e:
            logger.error(f"[TempGraphRAG] Error building temporary GraphRAG: {e}", exc_info=True)
            return None, None
    
    # ========================================================================
    # NODE IMPLEMENTATIONS (Placeholders for Phase 4.1)
    # ========================================================================
    
    async def _analyze_node(self, state: ReviewState) -> ReviewState:
        """
        Analysis node: Build comprehensive PR context using ContextBuilder
        
        This node:
        1. Converts file dicts to FileChange objects
        2. **NEW**: Builds temporary in-memory GraphRAG for PR files
        3. Builds complete PR context with graph + vector + analysis
        4. **NEW**: Merges temporary and permanent GraphRAG contexts
        5. Extracts risk level and metrics
        6. Prepares state for routing
        """
        logger.info(f"[ANALYZE] Starting analysis for PR #{state['pr_number']}")
        
        state["status"] = "analyzing"
        state["current_step"] = "context_building"
        state["started_at"] = datetime.utcnow().isoformat()
        
        try:
            if not self.context_builder:
                logger.error("ContextBuilder not initialized")
                state["errors"] = [{"message": "ContextBuilder not available", "step": "analyze"}]
                return state
            
            # ==================================================================
            # STEP 1: Build Temporary GraphRAG for PR Files
            # ==================================================================
            logger.info("[ANALYZE] Step 1: Building temporary in-memory GraphRAG")
            temp_graph, temp_vector = await self._build_temporary_graphrag(state["files"])
            
            # Create context merger
            context_merger = ContextMerger(temp_graph=temp_graph, temp_vector=temp_vector)
            
            # Log statistics
            stats = context_merger.get_statistics()
            logger.info(f"[ANALYZE] Temporary GraphRAG stats: {stats}")
            
            # ==================================================================
            # STEP 2: Convert File Dicts to FileChange Objects
            # ==================================================================
            file_changes = []
            for f in state["files"]:
                file_change = FileChange(
                    filename=f.get("filename", ""),
                    status=f.get("status", "modified"),
                    additions=f.get("additions", 0),
                    deletions=f.get("deletions", 0),
                    patch=f.get("patch"),
                    language=f.get("language")
                )
                file_changes.append(file_change)
            
            # ==================================================================
            # STEP 3: Build Comprehensive PR Context (Permanent + Temporary)
            # ==================================================================
            logger.info(f"[ANALYZE] Step 2: Building context for {len(file_changes)} files")
            
            # Extract base_ref and head_ref from state
            base_ref = state.get("base_ref", "main")
            head_ref = state.get("head_ref", "unknown")
            
            # Build PR context with permanent databases
            pr_context = await self.context_builder.build_pr_context(
                pr_number=state["pr_number"],
                repo_id=state["repo_id"],
                title=state["pr_title"],
                description=state.get("pr_description"),
                files=file_changes,
                base_ref=base_ref,
                head_ref=head_ref
            )
            
            # ==================================================================
            # STEP 4: Search Temporary Vectors for Similar Code
            # ==================================================================
            logger.info("[ANALYZE] Step 3: Searching temporary vectors for similar code")
            
            # For each file in temp graph, find similar code within the PR
            temp_similar_results = {}
            if temp_vector and len(temp_vector.vectors) > 0:
                for filename in temp_graph.files.keys():
                    file_node = temp_graph.files[filename]
                    file_similar = []
                    
                    # For each node in this file, find similar nodes
                    for node in file_node.nodes:
                        similar = temp_vector.find_similar_to_node(
                            node.id,
                            limit=3,
                            min_score=0.5  # Lower threshold for PR-internal similarities
                        )
                        if similar:
                            file_similar.extend(similar)
                    
                    if file_similar:
                        temp_similar_results[filename] = file_similar
                        logger.info(f"[ANALYZE] Found {len(file_similar)} similar code snippets in {filename} from temporary GraphRAG")
            
            # ==================================================================
            # STEP 5: Merge Temporary + Permanent Contexts
            # ==================================================================
            logger.info("[ANALYZE] Step 4: Merging temporary and permanent contexts")
            
            # Enhance file contexts with merged data
            for file_ctx in pr_context.files:
                # Get permanent context for this file
                permanent_context = {
                    "dependencies": [e.model_dump() for e in file_ctx.entities if hasattr(e, 'dependencies')],
                    "dependents": [],
                    "similar_code": [],  # Will be populated from entities
                    "imports": getattr(file_ctx, 'imports', []),
                    "file_dependencies": getattr(file_ctx, 'file_dependencies', [])
                }
                
                # Extract similar_code from entities
                for entity in file_ctx.entities:
                    if hasattr(entity, 'similar_code') and entity.similar_code:
                        permanent_context["similar_code"].extend(entity.similar_code)
                
                # Add temporary similar code results
                if file_ctx.filename in temp_similar_results:
                    permanent_context["similar_code"].extend(temp_similar_results[file_ctx.filename])
                
                # Merge with temporary context
                merged = context_merger.merge_file_context(
                    filename=file_ctx.filename,
                    permanent_context=permanent_context
                )
                
                # Store merged context in separate dict (can't add to Pydantic model)
                if "_merged_contexts" not in state:
                    state["_merged_contexts"] = {}
                # Convert dataclass to dict for JSON serialization
                state["_merged_contexts"][file_ctx.filename] = asdict(merged)
            
            # Check if GraphRAG data is available (permanent or temporary)
            merged_contexts = state.get("_merged_contexts", {})
            has_graphrag_data = any(
                len(f.entities) > 0 or len(f.dependencies) > 0 or 
                (f.filename in merged_contexts and merged_contexts[f.filename].get("temp_nodes_count", 0) > 0)
                for f in pr_context.files
            )
            
            if not has_graphrag_data:
                logger.warning(
                    f"⚠️ No GraphRAG data found for {state['repo_id']}. "
                    "Repository may not have been ingested yet. "
                    "Similar code and dependency analysis will be limited. "
                    f"Run POST /ingest with repo_url to enable full GraphRAG features."
                )
            elif temp_graph and len(temp_graph.files) > 0:
                logger.info(
                    f"✅ Using hybrid GraphRAG context: "
                    f"{len(temp_graph.files)} files with temporary analysis + permanent database"
                )
            
            # Store context in state (convert Pydantic model to dict)
            state["pr_context"] = pr_context.model_dump()
            
            # Store context merger for use in review nodes
            state["_context_merger"] = context_merger  # Internal, not serialized
            
            # Extract key metrics for routing
            state["risk_level"] = pr_context.risk_level
            state["requires_deep_review"] = pr_context.requires_deep_review
            
            # Prioritize files based on complexity and issues
            high_priority = []
            medium_priority = []
            low_priority = []
            
            for file_ctx in pr_context.files:
                if file_ctx.complexity_score >= self.config.high_priority_complexity:
                    high_priority.append(file_ctx.filename)
                elif file_ctx.issues_summary.get("critical", 0) + file_ctx.issues_summary.get("high", 0) >= self.config.high_priority_issues:
                    high_priority.append(file_ctx.filename)
                elif file_ctx.complexity_score >= 40:
                    medium_priority.append(file_ctx.filename)
                else:
                    low_priority.append(file_ctx.filename)
            
            state["high_priority_files"] = high_priority
            state["medium_priority_files"] = medium_priority
            state["low_priority_files"] = low_priority
            
            logger.info(f"Analysis complete: risk={pr_context.risk_level}, "
                       f"high_priority={len(high_priority)}, "
                       f"critical_issues={len(pr_context.critical_issues)}")
            
            state["messages"].append({
                "role": "system",
                "content": f"Context built: {pr_context.total_files} files analyzed, "
                          f"{len(pr_context.critical_issues)} critical issues found"
            })
            
        except Exception as e:
            logger.error(f"Error in analyze node: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [
                {"message": str(e), "step": "analyze", "timestamp": datetime.utcnow().isoformat()}
            ]
        
        return state
    
    async def _route_node(self, state: ReviewState) -> ReviewState:
        """
        Routing node: Determine review strategy and select Gemini model
        
        Strategy Selection:
        - quick: Small, low-risk PRs (< 3 files, < 100 additions)
        - standard: Typical PRs (< 10 files, < 500 additions)
        - deep: Large, complex, or high-risk PRs
        """
        logger.info(f"[ROUTE] Determining review strategy for PR #{state['pr_number']}")
        
        state["current_step"] = "routing"
        
        try:
            pr_context = state.get("pr_context", {})
            total_files = pr_context.get("total_files", len(state["files"]))
            total_additions = pr_context.get("total_additions", 0)
            risk_level = state.get("risk_level", "low")
            requires_deep = state.get("requires_deep_review", False)
            
            # Determine strategy
            if total_files <= self.config.quick_review_max_files and \
               total_additions <= self.config.quick_review_max_additions and \
               risk_level == "low" and \
               not requires_deep:
                strategy = "quick"
            
            elif total_files > self.config.standard_review_max_files or \
                 total_additions > self.config.standard_review_max_additions or \
                 risk_level in ["high", "critical"] or \
                 requires_deep:
                strategy = "deep"
            
            else:
                strategy = "standard"
            
            state["review_strategy"] = strategy
            
            # Select Gemini model based on strategy and PR characteristics
            selected_model = self.gemini_client.select_model(
                total_files=total_files,
                total_additions=total_additions,
                risk_level=risk_level,
                review_strategy=strategy
            )
            
            state["selected_model"] = selected_model
            
            logger.info(f"Route decision: strategy={strategy}, model={selected_model}, "
                       f"files={total_files}, additions={total_additions}, risk={risk_level}")
            
            state["messages"].append({
                "role": "system",
                "content": f"Review strategy: {strategy} using {selected_model}"
            })
            
        except Exception as e:
            logger.error(f"Error in route node: {e}", exc_info=True)
            # Default to standard review on error
            state["review_strategy"] = "standard"
            state["selected_model"] = self.config.flash_model
            state["errors"] = state.get("errors", []) + [
                {"message": str(e), "step": "route", "timestamp": datetime.utcnow().isoformat()}
            ]
        
        return state
    
    def _route_decision(self, state: ReviewState) -> str:
        """
        Conditional edge: Decide which review path to take
        
        Returns: "quick", "standard", "deep", or "error"
        """
        # Check for critical errors in analysis/routing
        errors = state.get("errors", [])
        if errors and any(e.get("step") in ["analyze", "route"] for e in errors):
            logger.error(f"[ROUTE_DECISION] Errors detected, routing to error handler")
            return "error"
        
        strategy = state.get("review_strategy", "standard")
        logger.info(f"[ROUTE_DECISION] Selected strategy: {strategy}")
        return strategy
    
    async def _review_quick_node(self, state: ReviewState) -> ReviewState:
        """
        Quick review for small, low-risk PRs
        
        Uses gemini-flash-lite with focused prompt for speed
        Reviews all files together in a single prompt
        """
        logger.info(f"[REVIEW_QUICK] Processing PR #{state['pr_number']}")
        state["status"] = "reviewing"
        state["current_step"] = "quick_review"
        
        try:
            pr_context = state.get("pr_context", {})
            
            # Build quick review prompt
            issues_summary = self._format_issues_summary(pr_context)
            files_summary = self._format_files_summary(pr_context, max_files=10)
            
            prompt = self.gemini_client.templates.QUICK_REVIEW_PROMPT.format(
                pr_title=state["pr_title"],
                total_files=pr_context.get("total_files", 0),
                additions=pr_context.get("total_additions", 0),
                deletions=pr_context.get("total_deletions", 0),
                description=state.get("pr_description", "No description provided"),
                files_summary=files_summary,
                issues_summary=issues_summary
            )
            
            model = state.get("selected_model", self.config.flash_lite_model)
            
            # Generate review
            review = await self.gemini_client.generate_review(
                model_name=model,
                prompt=prompt
            )
            
            # Store review
            state["overall_summary"] = review
            state["processed_files"] = pr_context.get("total_files", 0)
            
            logger.info(f"Quick review completed ({len(review)} chars)")
            
            state["messages"].append({
                "role": "assistant",
                "content": f"Quick review generated for {pr_context.get('total_files', 0)} files"
            })
            
        except Exception as e:
            logger.error(f"Error in quick review: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [
                {"message": str(e), "step": "review_quick", "timestamp": datetime.utcnow().isoformat()}
            ]
            state["overall_summary"] = "Unable to generate quick review due to an error."
        
        return state
    
    async def _review_standard_node(self, state: ReviewState) -> ReviewState:
        """
        Standard review for typical PRs
        
        Reviews files in batches, prioritizing high-risk files
        Generates individual file reviews then aggregates
        """
        logger.info(f"[REVIEW_STANDARD] Processing PR #{state['pr_number']}")
        state["status"] = "reviewing"
        state["current_step"] = "standard_review"
        
        try:
            pr_context = state.get("pr_context", {})
            files = pr_context.get("files", [])
            
            # Prioritize files for review
            high_priority = state.get("high_priority_files", [])
            medium_priority = state.get("medium_priority_files", [])
            low_priority = state.get("low_priority_files", [])
            
            prioritized_files = high_priority + medium_priority + low_priority
            
            # Review files in parallel for much better performance
            file_reviews = {}
            model = state.get("selected_model", self.config.flash_model)
            
            # Collect files to review
            review_tasks = []
            files_to_review = []
            for filename in prioritized_files[:self.config.max_files_per_batch * 2]:
                file_ctx = self._find_file_context(files, filename)
                if file_ctx:
                    files_to_review.append(filename)
                    review_tasks.append(self._review_single_file(
                        model=model,
                        file_ctx=file_ctx,
                        state=state
                    ))
            
            # Review all files in parallel
            logger.info(f"Reviewing {len(review_tasks)} files in parallel...")
            import asyncio
            reviews = await asyncio.gather(*review_tasks, return_exceptions=True)
            
            # Collect results
            for filename, review in zip(files_to_review, reviews):
                if isinstance(review, Exception):
                    logger.error(f"Error reviewing {filename}: {review}")
                    file_reviews[filename] = {
                        "filename": filename,
                        "summary": f"Error: {str(review)}",
                        "error": True
                    }
                else:
                    file_reviews[filename] = review
            
            state["file_reviews"] = file_reviews
            state["processed_files"] = len(file_reviews)
            
            logger.info(f"Standard review completed: {len(file_reviews)} files reviewed")
            
            state["messages"].append({
                "role": "assistant",
                "content": f"Reviewed {len(file_reviews)} files in standard mode"
            })
            
        except Exception as e:
            logger.error(f"Error in standard review: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [
                {"message": str(e), "step": "review_standard", "timestamp": datetime.utcnow().isoformat()}
            ]
        
        return state
    
    async def _review_deep_node(self, state: ReviewState) -> ReviewState:
        """
        Deep review for complex, high-risk PRs
        
        Uses gemini-pro for thorough analysis
        Reviews all files individually with full context
        Includes architecture and security deep dives
        """
        logger.info(f"[REVIEW_DEEP] Processing PR #{state['pr_number']}")
        state["status"] = "reviewing"
        state["current_step"] = "deep_review"
        
        try:
            pr_context = state.get("pr_context", {})
            files = pr_context.get("files", [])
            
            # Build comprehensive deep review prompt
            critical_issues = self._format_critical_issues(pr_context)
            high_issues = self._format_high_issues(pr_context)
            files_details = self._format_files_details(files)
            
            prompt = self.gemini_client.templates.DEEP_REVIEW_PROMPT.format(
                pr_title=state["pr_title"],
                description=state.get("pr_description", "No description provided"),
                total_files=pr_context.get("total_files", 0),
                additions=pr_context.get("total_additions", 0),
                deletions=pr_context.get("total_deletions", 0),
                languages=", ".join(pr_context.get("languages", [])),
                risk_level=pr_context.get("risk_level", "unknown").upper(),
                critical_issues=critical_issues,
                high_issues=high_issues,
                affected_callers=pr_context.get("affected_callers", 0),
                complexity_hotspots=self._format_complexity_hotspots(pr_context),
                coupling_files=self._format_coupling_files(pr_context),
                similar_code=self._format_similar_code(pr_context),
                files_details=files_details
            )
            
            model = state.get("selected_model", self.config.pro_model)
            
            # Generate deep review (may take longer)
            logger.info(f"Generating deep review with {model}...")
            review = await self.gemini_client.generate_review(
                model_name=model,
                prompt=prompt
            )
            
            state["overall_summary"] = review
            state["processed_files"] = len(files)
            
            logger.info(f"Deep review completed ({len(review)} chars)")
            
            state["messages"].append({
                "role": "assistant",
                "content": f"Deep review generated with comprehensive analysis"
            })
            
        except Exception as e:
            logger.error(f"Error in deep review: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [
                {"message": str(e), "step": "review_deep", "timestamp": datetime.utcnow().isoformat()}
            ]
            state["overall_summary"] = "Unable to generate deep review due to an error."
        
        return state
    
    async def _aggregate_node(self, state: ReviewState) -> ReviewState:
        """
        Aggregate results and create final review
        
        For quick/deep reviews: formats the single review
        For standard reviews: aggregates multiple file reviews into unified review
        """
        logger.info(f"[AGGREGATE] Aggregating review for PR #{state['pr_number']}")
        
        state["status"] = "aggregating"
        state["current_step"] = "aggregation"
        
        try:
            review_strategy = state.get("review_strategy", "standard")
            
            # If we already have overall_summary (quick/deep), just format it
            if state.get("overall_summary"):
                logger.info("Using existing overall summary")
                state["status"] = "completed"
                state["completed_at"] = datetime.utcnow().isoformat()
                return state
            
            # For standard review, aggregate file reviews
            file_reviews = state.get("file_reviews", {})
            
            if not file_reviews:
                logger.warning("No file reviews to aggregate")
                state["overall_summary"] = "No detailed reviews were generated."
                state["status"] = "completed"
                state["completed_at"] = datetime.utcnow().isoformat()
                return state
            
            # Build aggregation prompt
            pr_context = state.get("pr_context", {})
            file_reviews_text = "\n\n".join([
                f"### {filename}\n{review.get('summary', '')}" 
                for filename, review in file_reviews.items()
            ])
            
            prompt = self.gemini_client.templates.AGGREGATION_PROMPT.format(
                pr_title=state["pr_title"],
                files_count=len(file_reviews),
                total_issues=len(pr_context.get("critical_issues", [])) + len(pr_context.get("high_issues", [])),
                file_reviews=file_reviews_text,
                critical_count=len(pr_context.get("critical_issues", [])),
                high_count=len(pr_context.get("high_issues", [])),
                medium_count=len(pr_context.get("medium_issues", []))
            )
            
            model = state.get("selected_model", self.config.flash_model)
            
            # Generate aggregated review
            logger.info("Generating aggregated review...")
            aggregated_review = await self.gemini_client.generate_review(
                model_name=model,
                prompt=prompt
            )
            
            state["overall_summary"] = aggregated_review
            
            logger.info(f"Aggregation completed ({len(aggregated_review)} chars)")
            
            state["messages"].append({
                "role": "assistant",
                "content": "Final review aggregated from individual file reviews"
            })
            
        except Exception as e:
            logger.error(f"Error in aggregation: {e}", exc_info=True)
            state["errors"] = state.get("errors", []) + [
                {"message": str(e), "step": "aggregate", "timestamp": datetime.utcnow().isoformat()}
            ]
            # Fallback: concatenate file reviews
            file_reviews = state.get("file_reviews", {})
            if file_reviews:
                state["overall_summary"] = "\n\n".join([
                    f"**{filename}**\n{review.get('summary', '')}" 
                    for filename, review in file_reviews.items()
                ])
            else:
                state["overall_summary"] = "Unable to generate aggregated review."
        
        state["status"] = "completed"
        state["completed_at"] = datetime.utcnow().isoformat()
        
        return state
    
    async def _error_handler_node(self, state: ReviewState) -> ReviewState:
        """
        Handle errors and determine retry strategy
        
        Recovery strategies:
        1. Transient errors (rate limit, network): Retry
        2. Authentication errors: Fail immediately
        3. Review generation errors: Try fallback model
        4. Max retries exceeded: Generate basic review from context
        """
        logger.error(f"[ERROR_HANDLER] Processing error for PR #{state['pr_number']}")
        
        state["status"] = "error_recovery"
        retry_count = state.get("retry_count", 0)
        state["retry_count"] = retry_count + 1
        
        errors = state.get("errors", [])
        if not errors:
            logger.warning("Error handler called but no errors found")
            return state
        
        last_error = errors[-1]
        error_msg = last_error.get("message", "")
        error_step = last_error.get("step", "unknown")
        
        logger.info(f"Handling error from step '{error_step}': {error_msg}")
        
        # Check if error is recoverable
        if "rate limit" in error_msg.lower():
            logger.info("Rate limit error detected - will retry with delay")
            import asyncio
            await asyncio.sleep(self.config.retry_delay_seconds * (retry_count + 1))
            return state
        
        elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
            logger.error("Authentication error - cannot retry")
            state["status"] = "failed"
            state["overall_summary"] = "Unable to generate review: Authentication error with Gemini API"
            return state
        
        elif retry_count >= self.config.max_retries:
            logger.error(f"Max retries ({self.config.max_retries}) exceeded")
            # Generate fallback review from context
            state = await self._generate_fallback_review(state)
            return state
        
        else:
            logger.info(f"Recoverable error - will retry (attempt {retry_count + 1}/{self.config.max_retries})")
            return state
    
    def _error_decision(self, state: ReviewState) -> str:
        """
        Conditional edge: Decide whether to retry or fail
        
        Returns "retry" or "end"
        """
        retry_count = state.get("retry_count", 0)
        status = state.get("status", "")
        
        # If we already have a fallback review or explicitly failed, end
        if status == "failed" or state.get("overall_summary"):
            logger.info(f"[ERROR_DECISION] Ending workflow (status={status})")
            return "end"
        
        if retry_count < self.config.max_retries:
            logger.info(f"[ERROR_DECISION] Retrying (attempt {retry_count + 1}/{self.config.max_retries})")
            return "retry"
        else:
            logger.error(f"[ERROR_DECISION] Max retries exceeded, ending workflow")
            return "end"
    
    # ========================================================================
    # HELPER METHODS (Phase 4.4)
    # ========================================================================
    
    async def _review_single_file(self, model: str, file_ctx: Dict, state: ReviewState) -> Dict[str, Any]:
        """Review a single file with Gemini"""
        try:
            # Format issues summary from the dict of counts
            issues_summary = file_ctx.get("issues_summary", {})
            if isinstance(issues_summary, dict) and all(isinstance(k, str) for k in issues_summary.keys()):
                # issues_summary is {"critical": 0, "high": 1, "medium": 2, "low": 0}
                issue_parts = []
                for severity in ["critical", "high", "medium", "low"]:
                    count = issues_summary.get(severity, 0)
                    if count > 0:
                        issue_parts.append(f"{severity.upper()}: {count}")
                issues = ", ".join(issue_parts) if issue_parts else "No issues detected"
            else:
                issues = "No issues detected"
            
            # Handle dependencies - should be a list of strings
            deps = file_ctx.get("dependencies", [])
            if isinstance(deps, list) and deps:
                dependencies = "\n".join([f"- {dep}" for dep in deps[:10]])  # Limit to 10
            else:
                dependencies = "None"
            
            # Extract similar code from context (GraphRAG!)
            similar_code = self._format_similar_code_for_file(file_ctx, state)
            
            # Get file diff if available
            diff = self._get_file_diff(state["files"], file_ctx.get("filename", "unknown"))
            
            prompt = self.gemini_client.templates.FILE_REVIEW_PROMPT.format(
                filename=file_ctx.get("filename", "unknown"),
                language=file_ctx.get("language", "unknown"),
                additions=file_ctx.get("additions", 0),
                deletions=file_ctx.get("deletions", 0),
                issues=issues,
                similar_code=similar_code,
                dependencies=dependencies,
                diff=diff[:2000]  # Limit diff size
            )
            
            review_text = await self.gemini_client.generate_review(
                model_name=model,
                prompt=prompt
            )
            
            # Calculate issues_found from issues_summary dict
            issues_found = 0
            if isinstance(issues_summary, dict):
                issues_found = sum(v for v in issues_summary.values() if isinstance(v, int))
            
            return {
                "filename": file_ctx.get("filename", "unknown"),
                "summary": review_text,
                "language": file_ctx.get("language"),
                "issues_found": issues_found
            }
            
        except Exception as e:
            logger.error(f"Error reviewing file {file_ctx.get('filename', 'unknown')}: {e}", exc_info=True)
            return {
                "filename": file_ctx.get("filename", "unknown"),
                "summary": f"Unable to review this file: {str(e)}",
                "error": True
            }
    
    async def _generate_fallback_review(self, state: ReviewState) -> ReviewState:
        """Generate basic review from context when Gemini fails"""
        logger.info("Generating fallback review from context")
        
        try:
            pr_context = state.get("pr_context", {})
            
            # Build basic review from analyzed context
            critical_issues = pr_context.get("critical_issues", [])
            high_issues = pr_context.get("high_issues", [])
            recommendations = pr_context.get("recommendations", [])
            
            fallback_review = f"""# Code Review Summary

**Note:** This is an automated summary generated from static analysis. Full AI review could not be completed.

## Overview
- **Files Changed:** {pr_context.get('total_files', 0)}
- **Lines Changed:** +{pr_context.get('total_additions', 0)} -{pr_context.get('total_deletions', 0)}
- **Risk Level:** {pr_context.get('risk_level', 'unknown').upper()}

## Critical Issues ({len(critical_issues)})
"""
            
            for issue in critical_issues[:5]:
                fallback_review += f"\n- ⚠️ **{issue.get('file')}** (line {issue.get('line', '?')}): {issue.get('description', '')}"
            
            if len(critical_issues) > 5:
                fallback_review += f"\n- ... and {len(critical_issues) - 5} more"
            
            fallback_review += f"\n\n## High Priority Issues ({len(high_issues)})\n"
            
            for issue in high_issues[:5]:
                fallback_review += f"\n- ⚠️ **{issue.get('file')}**: {issue.get('description', '')}"
            
            if len(high_issues) > 5:
                fallback_review += f"\n- ... and {len(high_issues) - 5} more"
            
            fallback_review += "\n\n## Recommendations\n"
            
            for rec in recommendations:
                fallback_review += f"\n- {rec}"
            
            state["overall_summary"] = fallback_review
            state["status"] = "completed_with_fallback"
            state["completed_at"] = datetime.utcnow().isoformat()
            
            logger.info("Fallback review generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating fallback review: {e}")
            state["overall_summary"] = "Unable to generate review due to technical difficulties."
            state["status"] = "failed"
        
        return state
    
    def _format_issues_summary(self, pr_context: Dict) -> str:
        """Format issues summary for prompts"""
        critical = len(pr_context.get("critical_issues", []))
        high = len(pr_context.get("high_issues", []))
        medium = len(pr_context.get("medium_issues", []))
        
        if critical + high + medium == 0:
            return "No major issues detected by static analysis."
        
        return f"Critical: {critical} | High: {high} | Medium: {medium}"
    
    def _format_files_summary(self, pr_context: Dict, max_files: int = 10) -> str:
        """Format files summary for prompts"""
        files = pr_context.get("files", [])[:max_files]
        
        summary = []
        for f in files:
            summary.append(
                f"- **{f.get('filename')}** ({f.get('language', '?')}): "
                f"+{f.get('additions', 0)} -{f.get('deletions', 0)}"
            )
        
        if len(pr_context.get("files", [])) > max_files:
            summary.append(f"- ... and {len(pr_context.get('files', [])) - max_files} more files")
        
        return "\n".join(summary)
    
    def _format_critical_issues(self, pr_context: Dict) -> str:
        """Format critical issues for deep review prompt"""
        issues = pr_context.get("critical_issues", [])
        
        if not issues:
            return "No critical issues detected."
        
        formatted = []
        for issue in issues[:10]:
            formatted.append(
                f"- **{issue.get('file')}** (line {issue.get('line', '?')}): "
                f"{issue.get('description', '')}"
            )
        
        if len(issues) > 10:
            formatted.append(f"- ... and {len(issues) - 10} more critical issues")
        
        return "\n".join(formatted)
    
    def _format_high_issues(self, pr_context: Dict) -> str:
        """Format high priority issues"""
        issues = pr_context.get("high_issues", [])
        
        if not issues:
            return "No high priority issues detected."
        
        formatted = []
        for issue in issues[:10]:
            formatted.append(
                f"- **{issue.get('file')}**: {issue.get('description', '')}"
            )
        
        if len(issues) > 10:
            formatted.append(f"- ... and {len(issues) - 10} more")
        
        return "\n".join(formatted)
    
    def _format_files_details(self, files: List[Dict]) -> str:
        """Format detailed file information"""
        details = []
        
        for f in files[:15]:  # Limit to 15 files for deep review
            details.append(
                f"### {f.get('filename')}\n"
                f"- Language: {f.get('language', 'unknown')}\n"
                f"- Changes: +{f.get('additions', 0)} -{f.get('deletions', 0)}\n"
                f"- Complexity Score: {f.get('complexity_score', 0)}/100\n"
                f"- Issues: {sum(f.get('issues_summary', {}).values())}"
            )
        
        return "\n\n".join(details)
    
    def _format_complexity_hotspots(self, pr_context: Dict) -> str:
        """Format complexity hotspots"""
        hotspots = pr_context.get("complexity_hotspots", [])
        
        if not hotspots:
            return "None detected"
        
        return "\n".join([
            f"- {h.get('file')}: {h.get('function')} ({h.get('call_count')} calls)"
            for h in hotspots[:5]
        ])
    
    def _format_coupling_files(self, pr_context: Dict) -> str:
        """Format highly coupled files"""
        coupling = pr_context.get("high_coupling_files", [])
        
        if not coupling:
            return "None detected"
        
        return "\n".join([
            f"- {c.get('source')} ↔ {c.get('target')} ({c.get('connections')} connections)"
            for c in coupling[:5]
        ])
    
    def _format_similar_code(self, pr_context: Dict) -> str:
        """Format similar code patterns for overall PR review"""
        # This would come from vector search results
        return "See individual file reviews for similar code patterns"
    
    def _format_similar_code_for_file(self, file_ctx: Dict, state: Dict) -> str:
        """
        Format similar code from GraphRAG vector search for a single file
        Now uses merged context (temporary + permanent)
        """
        # First check for merged_context from state (new hybrid approach)
        merged_contexts = state.get("_merged_contexts", {})
        merged = merged_contexts.get(file_ctx.get("filename"))
        
        if merged:
            similar_code = merged.get("similar_code", [])
            if similar_code:
                parts = [f"\n🔍 Similar Code Analysis:"]
                for sim in similar_code[:5]:  # Top 5 matches
                    metadata = sim.get("metadata", {})
                    source = sim.get("source", "unknown")
                    source_label = "🆕 PR" if source == "temporary" else "📦 Main"
                    
                    parts.append(
                        f"  {source_label} {metadata.get('file', 'unknown')} "
                        f"(score: {sim.get('score', 0):.2f}, "
                        f"line {metadata.get('line', '?')}, "
                        f"type: {metadata.get('type', 'unknown')})"
                    )
                return "\n".join(parts)
        
        # Fallback to legacy entity-based similar code
        entities = file_ctx.get("entities", [])
        if not entities:
            return "None found"
        
        similar_parts = []
        for entity in entities[:3]:  # Top 3 entities with similar code
            similar = entity.get("similar_code", [])
            if similar:
                similar_parts.append(f"\nEntity: {entity.get('name')}")
                for sim in similar[:2]:  # Top 2 similar matches per entity
                    similar_parts.append(
                        f"  - Similar in {sim.get('file', 'unknown')} "
                        f"(score: {sim.get('score', 0):.2f}, line {sim.get('line', '?')})"
                    )
        
        if similar_parts:
            return "\n".join(similar_parts)
        return "None found"
    
    def _find_file_context(self, files: List[Dict], filename: str) -> Optional[Dict]:
        """Find file context by filename"""
        for f in files:
            if f.get("filename") == filename:
                return f
        return None
    
    def _get_file_diff(self, files: List[Dict], filename: str) -> str:
        """Get diff/patch for a file"""
        for f in files:
            if f.get("filename") == filename:
                return f.get("patch", "No diff available")
        return "No diff available"
    
    # ========================================================================
    # WORKFLOW EXECUTION
    # ========================================================================
    
    async def run_review(
        self,
        pr_number: int,
        repo_id: str,
        pr_title: str,
        pr_description: Optional[str],
        files: List[Dict[str, Any]],
        base_ref: str = "main",
        head_ref: str = "unknown"
    ) -> Dict[str, Any]:
        """
        Execute the complete review workflow
        
        Args:
            pr_number: GitHub PR number
            repo_id: Repository identifier
            pr_title: PR title
            pr_description: PR description
            files: List of changed files
            base_ref: Base branch name
            head_ref: Head branch name
        
        Returns:
            Final state with review results
        """
        if not self.compiled:
            self.compile()
        
        # Initialize state
        initial_state: ReviewState = {
            "pr_number": pr_number,
            "repo_id": repo_id,
            "pr_title": pr_title,
            "pr_description": pr_description,
            "base_ref": base_ref,
            "head_ref": head_ref,
            "files": files,
            "status": "queued",
            "messages": [],
            "total_files": len(files),
            "processed_files": 0,
            "retry_count": 0,
            "errors": []
        }
        
        logger.info(f"Starting review workflow for PR #{pr_number} in {repo_id}")
        logger.info(f"PR: {pr_title}")
        logger.info(f"Files: {len(files)}")
        
        try:
            # Execute workflow
            final_state = await self.compiled.ainvoke(initial_state)
            
            logger.info(f"Workflow completed with status: {final_state.get('status')}")
            return final_state
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return {
                **initial_state,
                "status": "failed",
                "errors": [{"message": str(e), "timestamp": datetime.utcnow().isoformat()}]
            }
    
    async def stream_review(
        self,
        pr_number: int,
        repo_id: str,
        pr_title: str,
        pr_description: Optional[str],
        files: List[Dict[str, Any]],
        base_ref: str = "main",
        head_ref: str = "unknown"
    ):
        """
        Stream the review workflow execution
        Yields state updates as the workflow progresses
        """
        if not self.compiled:
            self.compile()
        
        initial_state: ReviewState = {
            "pr_number": pr_number,
            "repo_id": repo_id,
            "pr_title": pr_title,
            "pr_description": pr_description,
            "base_ref": base_ref,
            "head_ref": head_ref,
            "files": files,
            "status": "queued",
            "messages": [],
            "total_files": len(files),
            "processed_files": 0,
            "retry_count": 0,
            "errors": []
        }
        
        logger.info(f"Starting streaming review for PR #{pr_number}")
        
        try:
            async for state in self.compiled.astream(initial_state):
                yield state
        except Exception as e:
            logger.error(f"Streaming workflow failed: {e}", exc_info=True)
            yield {
                **initial_state,
                "status": "failed",
                "errors": [{"message": str(e), "timestamp": datetime.utcnow().isoformat()}]
            }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_review_workflow(
    config: Optional[WorkflowConfig] = None,
    context_builder: Optional[ContextBuilder] = None,
    gemini_client: Optional[GeminiClient] = None
) -> CodeReviewWorkflow:
    """
    Factory function to create a configured workflow
    
    Args:
        config: Workflow configuration
        context_builder: ContextBuilder instance (from Phase 3)
        gemini_client: GeminiClient instance for Gemini API
    
    Returns:
        Compiled CodeReviewWorkflow ready for execution
    """
    workflow = CodeReviewWorkflow(config, context_builder, gemini_client)
    workflow.compile()
    return workflow
