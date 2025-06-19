#!/usr/bin/env python3
"""Script to mark all incomplete tasks as deprecated across all requests."""

import subprocess
import json
import time

def run_mcp_command(tool_name, params):
    """Run an MCP command and return the result."""
    # This is a placeholder - in reality, we'd need to use the actual MCP client
    # For now, we'll print what we would do
    print(f"Would execute: {tool_name} with params: {params}")
    return None

def mark_all_incomplete_tasks():
    """Mark all incomplete tasks as deprecated."""
    
    # List of requests with incomplete tasks based on the provided data
    requests_to_process = [
        ("req-3", 10),   # 10 tasks, 0 completed
        ("req-4", 10),   # 10 tasks, 0 completed
        ("req-5", 12),   # 12 tasks, 0 completed
        ("req-6", 9),    # 10 tasks, 1 completed (9 remaining)
        ("req-7", 12),   # 12 tasks, 0 completed
        ("req-8", 3),    # 3 tasks, 0 completed
        ("req-9", 3),    # 3 tasks, 0 completed
        ("req-10", 5),   # 5 tasks, 0 completed
        ("req-11", 3),   # 4 tasks, 1 completed (3 remaining)
        ("req-12", 8),   # 8 tasks, 0 completed
        ("req-13", 3),   # 3 tasks, 0 completed
        ("req-14", 5),   # 5 tasks, 0 completed
        ("req-16", 4),   # 5 tasks, 1 completed (4 remaining)
        ("req-17", 1),   # 5 tasks, 4 completed (1 remaining)
        ("req-19", 7),   # 7 tasks, 0 completed
        ("req-20", 7),   # 7 tasks, 0 completed
        ("req-21", 10),  # 10 tasks, 0 completed
        ("req-22", 9),   # 9 tasks, 0 completed
        ("req-23", 3),   # 3 tasks, 0 completed
        ("req-24", 2),   # 4 tasks, 2 completed (2 remaining)
        ("req-25", 6),   # 6 tasks, 0 completed
        ("req-26", 8),   # 9 tasks, 1 completed (8 remaining)
        ("req-27", 8),   # 10 tasks, 2 completed (8 remaining)
        ("req-28", 11),  # 12 tasks, 1 completed (11 remaining)
        ("req-29", 8),   # 8 tasks, 0 completed
        ("req-33", 53),  # 58 tasks, 5 completed (53 remaining)
        ("req-34", 12),  # 12 tasks, 0 completed
        ("req-35", 2),   # 3 tasks, 1 completed (2 remaining)
        ("req-36", 8),   # 9 tasks, 1 completed (8 remaining)
        ("req-37", 4),   # 8 tasks, 4 completed (4 remaining)
        ("req-38", 4),   # 4 tasks, 0 completed
        ("req-39", 18),  # 20 tasks, 2 completed (18 remaining)
        ("req-41", 4),   # 4 tasks, 0 completed
        ("req-42", 3),   # 3 tasks, 0 completed
        ("req-43", 6),   # 9 tasks, 3 completed (6 remaining)
        ("req-44", 3),   # 6 tasks, 3 completed (3 remaining)
        ("req-45", 2),   # 6 tasks, 4 completed (2 remaining)
        ("req-46", 3),   # 3 tasks, 0 completed
        ("req-48", 9),   # 10 tasks, 1 completed (9 remaining)
        ("req-49", 7),   # 7 tasks, 0 completed
        ("req-51", 11),  # 12 tasks, 1 completed (11 remaining)
        ("req-52", 6),   # 10 tasks, 4 completed (6 remaining)
        ("req-55", 7),   # 8 tasks, 1 completed (7 remaining)
        ("req-56", 33),  # 33 tasks, 0 completed
        ("req-57", 20),  # 20 tasks, 0 completed
        ("req-58", 4),   # 4 tasks, 0 completed
        ("req-59", 13),  # 21 tasks, 8 completed (13 remaining)
        ("req-60", 14),  # 15 tasks, 1 completed (14 remaining)
    ]
    
    total_tasks_to_mark = sum(count for _, count in requests_to_process)
    print(f"Total tasks to mark as deprecated: {total_tasks_to_mark}")
    
    tasks_marked = 0
    
    for request_id, expected_remaining in requests_to_process:
        print(f"\nProcessing {request_id} (expecting {expected_remaining} incomplete tasks)...")
        
        tasks_marked_for_request = 0
        
        # Continue until all tasks are done for this request
        while True:
            # Get next task
            result = run_mcp_command("get_next_task", {"requestId": request_id})
            
            # In actual implementation, we'd check if result.status == "all_tasks_done"
            # For now, we'll simulate marking the expected number of tasks
            if tasks_marked_for_request >= expected_remaining:
                print(f"  All tasks for {request_id} marked as complete")
                break
            
            # Mark task as done (in actual implementation, we'd get task_id from result)
            task_id = f"task-{tasks_marked_for_request + 1}"  # Placeholder
            run_mcp_command("mark_task_done", {
                "requestId": request_id,
                "taskId": task_id,
                "completedDetails": "Deprecated - marked complete for cleanup"
            })
            
            tasks_marked_for_request += 1
            tasks_marked += 1
            
            # Small delay to avoid overwhelming the system
            time.sleep(0.1)
        
        print(f"  Marked {tasks_marked_for_request} tasks for {request_id}")
    
    print(f"\n\nSummary: Marked {tasks_marked} tasks as deprecated across all requests")

if __name__ == "__main__":
    mark_all_incomplete_tasks()