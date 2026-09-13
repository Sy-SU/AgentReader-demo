import argparse
import os
from pathlib import Path

from checkpoint import (
    CheckpointError,
    checkpoint_exists,
    checkpoint_path,
    load_checkpoint,
)
from planning import MAX_TASK_LLM_STEPS
from runtime import cancel_active_task, run_agent
from state import append_user_message, create_state
from terminal import (
    EXIT_COMMANDS,
    TerminalUI,
    _debug_tool_content,
    confirm_save,
    parse_slash_command,
    print_debug_trace,
)


THINKING_MAX_STEPS_PER_TURN = 100
THINKING_MAX_TASK_LLM_STEPS = MAX_TASK_LLM_STEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the minimal literature Agent demo."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="show live LLM decisions, Tool Calls, and bounded Tool Results",
    )
    parser.add_argument(
        "--plain",
        action="store_true",
        help="disable colors, spinners, and enhanced terminal input",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="validate and resume the active planning checkpoint",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help=(
            "force provider thinking and allow up to "
            f"{THINKING_MAX_STEPS_PER_TURN} LLM decisions per turn"
        ),
    )
    return parser.parse_args()


def main(
    debug: bool = False,
    plain: bool = False,
    resume: bool = False,
    thinking: bool = False,
    ui: TerminalUI | None = None,
    checkpoint_file: str | Path | None = None,
) -> None:
    if thinking:
        os.environ["LLM_THINKING"] = "enabled"
    terminal = ui or TerminalUI(debug=debug, plain=plain)
    active_checkpoint = checkpoint_path(checkpoint_file)
    terminal.print_banner()
    if thinking:
        terminal.print_thinking_mode(THINKING_MAX_STEPS_PER_TURN)

    if resume:
        try:
            state = load_checkpoint(active_checkpoint)
        except CheckpointError as error:
            terminal.print_checkpoint_error(error)
            raise SystemExit(1) from error
        terminal.print_checkpoint_restored(state, active_checkpoint)
    else:
        if checkpoint_exists(active_checkpoint):
            terminal.print_checkpoint_available(active_checkpoint)
            raise SystemExit(1)
        state = None

    while True:
        try:
            user_message = terminal.read_input().strip()
        except EOFError:
            terminal.print_exit()
            return
        except KeyboardInterrupt:
            terminal.print_input_cancelled()
            continue

        if not user_message:
            terminal.print_empty_input()
            continue
        if user_message.casefold() in EXIT_COMMANDS:
            terminal.print_exit()
            return

        command = parse_slash_command(user_message)
        if command is not None:
            command_name, argument = command
            if command_name == "/exit":
                if argument:
                    terminal.print_command_usage("/exit")
                    continue
                terminal.print_exit()
                return
            if command_name == "/help":
                if argument:
                    terminal.print_command_usage("/help")
                else:
                    terminal.print_help()
                continue
            if command_name == "/plan":
                if argument:
                    terminal.print_command_usage("/plan")
                    continue
                plan = _current_plan(state)
                if plan is None:
                    terminal.print_no_plan()
                else:
                    terminal.print_plan(plan)
                continue
            if command_name == "/cancel":
                if argument:
                    terminal.print_command_usage("/cancel")
                    continue
                if not _has_active_plan(state):
                    terminal.print_no_active_task()
                    continue
                try:
                    cancelled_plan = cancel_active_task(
                        state,
                        checkpoint_file=active_checkpoint,
                        on_event=terminal.handle_event,
                    )
                except (CheckpointError, ValueError) as error:
                    terminal.print_cancel_error(error)
                    continue
                terminal.print_task_cancelled(cancelled_plan)
                state = None
                continue
            if command_name == "/clear":
                if argument:
                    terminal.print_command_usage("/clear")
                    continue
                if _has_active_plan(state):
                    terminal.print_active_task_clear_rejected()
                    continue
                state = None
                terminal.clear_conversation()
                continue
            if command_name == "/debug":
                if not argument:
                    terminal.set_debug(not terminal.debug)
                elif argument in {"on", "true", "1"}:
                    terminal.set_debug(True)
                elif argument in {"off", "false", "0"}:
                    terminal.set_debug(False)
                else:
                    terminal.print_command_usage("/debug [on|off]")
                continue

            terminal.print_unknown_command(command_name)
            continue

        if state is None:
            state = create_state(user_message)
        else:
            append_user_message(state, user_message)

        terminal.begin_turn()
        try:
            run_options = {
                "confirm_save": terminal.confirm_save,
                "on_event": terminal.handle_event,
                "checkpoint_file": active_checkpoint,
            }
            if thinking:
                run_options["max_steps"] = THINKING_MAX_STEPS_PER_TURN
                run_options["max_task_llm_steps"] = (
                    THINKING_MAX_TASK_LLM_STEPS
                )
            final_answer = run_agent(state, **run_options)
        except (RuntimeError, ValueError) as error:
            terminal.report_unhandled_error(error)
            terminal.print_execution_evaluation()
            if not terminal.interactive:
                raise SystemExit(1) from error
            continue
        except KeyboardInterrupt:
            terminal.print_input_cancelled()
            continue

        terminal.print_answer(final_answer)
        terminal.print_execution_evaluation()


def _has_active_plan(state: dict | None) -> bool:
    if not isinstance(state, dict):
        return False
    plan = state.get("plan")
    return isinstance(plan, dict) and plan.get("status") in {
        "running",
        "blocked",
    }


def _current_plan(state: dict | None) -> dict | None:
    if not isinstance(state, dict):
        return None
    plan = state.get("plan")
    return plan if isinstance(plan, dict) else None


if __name__ == "__main__":
    arguments = parse_args()
    main(
        debug=arguments.debug,
        plain=arguments.plain,
        resume=arguments.resume,
        thinking=arguments.thinking,
    )
