from golden_agent.tools import bash_path, begin_session, read_file, run_command


def write_tmp(tmp_path, name, text):
    target = tmp_path / name
    target.write_text(text, encoding="utf-8")
    begin_session(tmp_path)
    return target


def test_read_file_whole_small_file(tmp_path):
    write_tmp(tmp_path, "small.txt", "alpha\nbeta\n")

    assert read_file("small.txt") == "alpha\nbeta"


def test_read_file_offset_starts_at_given_line(tmp_path):
    lines = [f"line {i}" for i in range(1, 11)]
    write_tmp(tmp_path, "numbered.txt", "\n".join(lines))

    result = read_file("numbered.txt", offset=8)

    assert result.splitlines() == ["line 8", "line 9", "line 10"]


def test_read_file_caps_and_pages_through_entire_file(monkeypatch, tmp_path):
    import golden_agent.tools as tools

    monkeypatch.setattr(tools, "MAX_READ_CHARS", 200)
    lines = [f"row-{i:03d} padding padding" for i in range(1, 101)]
    write_tmp(tmp_path, "big.txt", "\n".join(lines))

    collected: list[str] = []
    offset = 1
    for _ in range(50):
        result = read_file("big.txt", offset=offset)
        body, _, footer = result.rpartition("\n[")
        collected.extend(body.splitlines())
        next_offset = footer.rstrip("]").rsplit("=", 1)[-1]
        if not next_offset.isdigit():
            collected.extend(footer.splitlines())
            break
        offset = int(next_offset)

    assert collected == lines


def test_bash_executable_is_located():
    assert bash_path() is not None


def test_run_command_captures_stdout():
    result = run_command("echo golden-agent-bash-test")

    assert "golden-agent-bash-test" in result
    assert result.startswith("[exit 0]")


def test_run_command_reports_nonzero_exit_code():
    result = run_command("echo oops >&2; exit 3")

    assert "[exit 3]" in result
    assert "oops" in result


def test_run_command_runs_from_launch_dir(tmp_path):
    from golden_agent.tools import begin_session

    begin_session(tmp_path)
    try:
        result = run_command("touch marker.txt")
    finally:
        begin_session()

    assert "[exit 0]" in result
    assert (tmp_path / "marker.txt").exists()


def test_unknown_tool_error_lists_valid_tools():
    from golden_agent.tools import execute_tool

    result = execute_tool("search_web", {})

    assert "unknown tool 'search_web'" in result
    for name in ("read_file", "write_file", "edit_file", "bash", "web_search", "web_fetch"):
        assert name in result
