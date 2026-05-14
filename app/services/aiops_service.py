"""
Generic Plan-Execute-Replan service
Implemented based on the official LangGraph tutorial
"""

from typing import AsyncGenerator, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from loguru import logger

from app.agent.aiops import PlanExecuteState, planner, executor, replanner


# Node name constants
NODE_PLANNER = "planner"
NODE_EXECUTOR = "executor"
NODE_REPLANNER = "replanner"


class AIOpsService:
    """Generic Plan-Execute-Replan service"""

    def __init__(self):
        """Initialize the service"""
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        logger.info("Plan-Execute-Replan Service initialized")

    def _build_graph(self):
        """Build the Plan-Execute-Replan workflow"""
        logger.info("Building workflow graph...")

        workflow = StateGraph(PlanExecuteState)

        workflow.add_node(NODE_PLANNER, planner)
        workflow.add_node(NODE_EXECUTOR, executor)
        workflow.add_node(NODE_REPLANNER, replanner)

        workflow.set_entry_point(NODE_PLANNER)

        workflow.add_edge(NODE_PLANNER, NODE_EXECUTOR)
        workflow.add_edge(NODE_EXECUTOR, NODE_REPLANNER)

        def should_continue(state: PlanExecuteState) -> str:
            """Determine whether to continue execution"""
            if state.get("response"):
                logger.info("Final response generated, ending workflow")
                return END

            plan = state.get("plan", [])
            if plan:
                logger.info(f"Continuing execution, {len(plan)} step(s) remaining")
                return NODE_EXECUTOR

            logger.info("Plan exhausted, generating final response")
            return END

        workflow.add_conditional_edges(
            NODE_REPLANNER,
            should_continue,
            {
                NODE_EXECUTOR: NODE_EXECUTOR,
                END: END
            }
        )

        compiled_graph = workflow.compile(checkpointer=self.checkpointer)

        logger.info("Workflow graph built successfully")
        return compiled_graph

    async def execute(
        self,
        user_input: str,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Execute the Plan-Execute-Replan workflow

        Args:
            user_input: Task description from the user
            session_id: Session ID

        Yields:
            Dict[str, Any]: Streaming events
        """
        logger.info(f"[Session {session_id}] Starting task execution: {user_input}")

        try:
            initial_state: PlanExecuteState = {
                "input": user_input,
                "plan": [],
                "past_steps": [],
                "response": ""
            }

            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for event in self.graph.astream(
                input=initial_state,
                config=config_dict,
                stream_mode="updates"
            ):
                for node_name, node_output in event.items():
                    logger.info(f"Node '{node_name}' emitted an event")

                    if node_name == NODE_PLANNER:
                        yield self._format_planner_event(node_output)

                    elif node_name == NODE_EXECUTOR:
                        yield self._format_executor_event(node_output)

                    elif node_name == NODE_REPLANNER:
                        yield self._format_replanner_event(node_output)

            final_state = self.graph.get_state(config_dict)
            final_response = ""

            if final_state and final_state.values:
                final_response = final_state.values.get("response", "")

            yield {
                "type": "complete",
                "stage": "complete",
                "message": "Task execution completed",
                "response": final_response
            }

            logger.info(f"[Session {session_id}] Task execution completed")

        except Exception as e:
            logger.error(f"[Session {session_id}] Task execution failed: {e}", exc_info=True)
            yield {
                "type": "error",
                "stage": "error",
                "message": f"Task execution error: {str(e)}"
            }

    async def diagnose(
        self,
        session_id: str = "default"
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        AIOps diagnosis interface (legacy compatibility)

        Args:
            session_id: Session ID

        Yields:
            Dict[str, Any]: Streaming events from the diagnosis process
        """
        from textwrap import dedent
        aiops_task = dedent("""Diagnose whether there are any active alerts in the current system. If alerts exist, perform a detailed analysis of the root cause and generate a diagnostic report in the following format:
                ```
                # Alert Analysis Report

                ---

                ## 📋 Active Alert List

                | Alert Name | Severity | Target Service | First Triggered | Last Triggered | Status |
                |------------|----------|----------------|-----------------|----------------|--------|
                | [Alert 1]  | [Level]  | [Service Name] | [Time]          | [Time]         | Active |
                | [Alert 2]  | [Level]  | [Service Name] | [Time]          | [Time]         | Active |

                ---

                ## 🔍 Root Cause Analysis 1 - [Alert Name]

                ### Alert Details
                - **Severity**: [Level]
                - **Affected Service**: [Service Name]
                - **Duration**: [X minutes]

                ### Symptom Description
                [Describe symptoms based on monitoring metrics]

                ### Log Evidence
                [Cite key log entries found during investigation]

                ### Root Cause Conclusion
                [Root cause derived from the evidence]

                ---

                ## 🛠️ Remediation Plan 1 - [Alert Name]

                ### Investigation Steps Performed
                1. [Step 1]
                2. [Step 2]

                ### Recommended Actions
                [Provide specific remediation recommendations]

                ### Expected Outcome
                [Describe the expected result after remediation]

                ---

                ## 🔍 Root Cause Analysis 2 - [Alert Name]
                [Repeat the above format if there is a second alert]

                ---

                ## 📊 Summary

                ### Overall Assessment
                [Summarize the overall situation of all alerts]

                ### Key Findings
                - [Finding 1]
                - [Finding 2]

                ### Follow-up Recommendations
                1. [Recommendation 1]
                2. [Recommendation 2]

                ### Risk Assessment
                [Assess the current risk level and impact scope]
                ```

                **Important Notes**:
                - The final output must be plain Markdown text, without any JSON structure
                - All content must be based on real data retrieved from tools; do not fabricate information
                - If a step fails, state it honestly in the summary rather than skipping it""")

        async for event in self.execute(aiops_task, session_id):
            if event.get("type") == "complete":
                yield {
                    "type": "complete",
                    "stage": "diagnosis_complete",
                    "message": "Diagnosis workflow completed",
                    "diagnosis": {
                        "status": "completed",
                        "report": event.get("response", "")
                    }
                }
            else:
                yield event

    def _format_planner_event(self, state: Dict | None) -> Dict:
        """Format a Planner node event"""
        if not state:
            return {
                "type": "status",
                "stage": "planner",
                "message": "Planner node running"
            }

        plan = state.get("plan", [])

        return {
            "type": "plan",
            "stage": "plan_created",
            "message": f"Execution plan created with {len(plan)} step(s)",
            "plan": plan
        }

    def _format_executor_event(self, state: Dict | None) -> Dict:
        """Format an Executor node event"""
        if not state:
            return {
                "type": "status",
                "stage": "executor",
                "message": "Executor node running"
            }

        plan = state.get("plan", [])
        past_steps = state.get("past_steps", [])

        if past_steps:
            last_step, _ = past_steps[-1]
            return {
                "type": "step_complete",
                "stage": "step_executed",
                "message": f"Step completed ({len(past_steps)}/{len(past_steps) + len(plan)})",
                "current_step": last_step,
                "remaining_steps": len(plan)
            }
        else:
            return {
                "type": "status",
                "stage": "executor",
                "message": "Starting step execution"
            }

    def _format_replanner_event(self, state: Dict | None) -> Dict:
        """Format a Replanner node event"""
        if not state:
            return {
                "type": "status",
                "stage": "replanner",
                "message": "Replanner node running"
            }

        response = state.get("response", "")
        plan = state.get("plan", [])

        if response:
            return {
                "type": "report",
                "stage": "final_report",
                "message": "Final report generated",
                "report": response
            }
        else:
            return {
                "type": "status",
                "stage": "replanner",
                "message": f"Evaluation complete, {'continuing with remaining steps' if plan else 'preparing final response'}",
                "remaining_steps": len(plan)
            }


# Global singleton
aiops_service = AIOpsService()
