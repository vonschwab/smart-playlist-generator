from src.playlist_gui.utils.bounded_buffer import BoundedBuffer


def test_bounded_buffer_drops_oldest_by_count():
    buf = BoundedBuffer(max_events=3, max_bytes=1000)
    for i in range(5):
        buf.append(f"item-{i}")
    assert len(buf) == 3
    assert buf.items() == ["item-2", "item-3", "item-4"]


def test_bounded_buffer_drops_by_bytes():
    buf = BoundedBuffer(max_events=100, max_bytes=10)
    buf.append("1234567890")  # 10 bytes
    buf.append("abc")  # pushes over byte limit; should drop oldest
    assert buf.items()[-1] == "abc"
    assert "1234567890" not in buf.items()
