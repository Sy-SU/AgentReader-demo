import argparse

from runtime import run_agent
from state import append_user_message, create_state
from terminal import (
    EXIT_COMMANDS,
    TerminalUI,
    _debug_tool_content,
    confirm_save,
    parse_slash_command,
    print_debug_trace,
)


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
    return parser.parse_args()


def main(
    debug: bool = False,
    plain: bool = False,
    ui: TerminalUI | None = None,
) -> None:
    terminal = ui or TerminalUI(debug=debug, plain=plain)
    state = None
    terminal.print_banner()

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
            if command_name == "/clear":
                if argument:
                    terminal.print_command_usage("/clear")
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
            final_answer = run_agent(
                state,
                confirm_save=terminal.confirm_save,
                on_event=terminal.handle_event,
            )
        except (RuntimeError, ValueError) as error:
            terminal.report_unhandled_error(error)
            if not terminal.interactive:
                raise SystemExit(1) from error
            continue
        except KeyboardInterrupt:
            terminal.print_input_cancelled()
            continue

        terminal.print_answer(final_answer)


if __name__ == "__main__":
    arguments = parse_args()
    main(debug=arguments.debug, plain=arguments.plain)
