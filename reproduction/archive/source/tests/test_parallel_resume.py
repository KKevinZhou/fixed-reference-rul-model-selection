import os

from resume_cap_sweep_parallel import process_alive


def test_process_alive_reports_current_and_impossible_pid():
    assert process_alive(os.getpid())
    assert not process_alive(2_147_483_647)
