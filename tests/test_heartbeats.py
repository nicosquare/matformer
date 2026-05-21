import io
import json


class FakeClock:
    def __init__(self, *values):
        self.values = list(values)
        self.last_value = values[-1] if values else 0.0

    def __call__(self):
        if self.values:
            self.last_value = self.values.pop(0)
        return self.last_value


def test_heartbeat_writer_emits_stdout_and_jsonl_schema(tmp_path):
    from utils.heartbeats import HeartbeatWriter

    stdout = io.StringIO()
    clock = FakeClock(1000.0, 1060.0)
    writer = HeartbeatWriter(
        output_dir=tmp_path,
        run_id="dmodel256-pilot-comparison-001",
        rank=0,
        world_size=2,
        stdout=stdout,
        time_fn=clock,
    )

    heartbeat_path = writer.emit(
        event_type="heartbeat",
        stage="training",
        step=10,
        derived_max_steps=100,
        tokens_seen=81_920,
        content_tokens_seen=74_250,
        token_budget=1_000_000,
        latest_loss=1.25,
        tokens_per_second=512.0,
        peak_gpu_memory_bytes=123_456,
        eta_seconds=120.0,
    )

    assert heartbeat_path == tmp_path / "heartbeats.jsonl"
    events = [
        json.loads(line)
        for line in heartbeat_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(events) == 1

    event = events[0]
    required_fields = {
        "event_type",
        "run_id",
        "stage",
        "rank",
        "world_size",
        "timestamp",
        "elapsed_seconds",
        "step",
        "derived_max_steps",
        "tokens_seen",
        "content_tokens_seen",
        "token_budget",
        "latest_loss",
        "tokens_per_second",
        "peak_gpu_memory_bytes",
        "eta_seconds",
    }
    assert required_fields <= set(event)
    assert event["event_type"] == "heartbeat"
    assert event["run_id"] == "dmodel256-pilot-comparison-001"
    assert event["stage"] == "training"
    assert event["rank"] == 0
    assert event["world_size"] == 2
    assert event["elapsed_seconds"] == 60.0
    assert event["step"] == 10
    assert event["derived_max_steps"] == 100
    assert event["tokens_seen"] == 81_920
    assert event["content_tokens_seen"] == 74_250
    assert event["token_budget"] == 1_000_000

    stdout_line = stdout.getvalue()
    assert "heartbeat" in stdout_line
    assert "stage=training" in stdout_line
    assert "rank=0/2" in stdout_line
    assert "step=10/100" in stdout_line
    assert "tokens=81920/1000000" in stdout_line
    assert "content_tokens=74250" in stdout_line


def test_heartbeat_writer_accepts_extra_stage_fields(tmp_path):
    from utils.heartbeats import HeartbeatWriter

    stdout = io.StringIO()
    writer = HeartbeatWriter(
        output_dir=tmp_path,
        run_id="dmodel256-nested-random-001",
        rank=0,
        world_size=4,
        stdout=stdout,
        time_fn=FakeClock(1000.0, 1001.0),
    )

    writer.stage_start("checkpointing", checkpoint_status="best_eval")

    event = json.loads((tmp_path / "heartbeats.jsonl").read_text(encoding="utf-8"))
    assert event["event_type"] == "stage_start"
    assert event["stage"] == "checkpointing"
    assert event["checkpoint_status"] == "best_eval"


def test_heartbeat_cadence_emits_on_step_or_elapsed_interval():
    from utils.heartbeats import HeartbeatCadence

    cadence = HeartbeatCadence(step_interval=10, time_interval_seconds=60)

    assert cadence.should_emit(step=0, now=0.0) is True
    cadence.mark_emitted(step=0, now=0.0)

    assert cadence.should_emit(step=9, now=59.0) is False
    assert cadence.should_emit(step=10, now=30.0) is True

    cadence.mark_emitted(step=10, now=30.0)

    assert cadence.should_emit(step=11, now=89.0) is False
    assert cadence.should_emit(step=11, now=90.0) is True

    cadence.mark_emitted(step=None, now=90.0)

    assert cadence.should_emit(step=None, now=149.0) is False
    assert cadence.should_emit(step=None, now=150.0) is True
