from src.playlist_gui.utils.redaction import redact_text


def test_redact_basic_tokens():
    text = "api_key=SECRET token: ABC password=foo Authorization: Bearer XYZ"
    redacted = redact_text(text)
    assert "SECRET" not in redacted
    assert "ABC" not in redacted
    assert "foo" not in redacted
    assert "XYZ" not in redacted


def test_redact_url_and_auth_header():
    text = "GET /?token=abcd&x=1\nAuthorization: Bearer SUPERSECRET\nkey=value"
    redacted = redact_text(text)
    assert "abcd" not in redacted
    assert "SUPERSECRET" not in redacted
    assert "value" not in redacted


def test_redact_traceback_like_dump():
    text = "Error: {'token':'abc', 'password':'secret', 'url':'http://x?api_key=Y'}"
    redacted = redact_text(text)
    assert "abc" not in redacted
    assert "secret" not in redacted
    assert "api_key" in redacted


def test_redact_url_query():
    text = "http://example.com?token=ABC123&x=1"
    redacted = redact_text(text)
    assert "ABC123" not in redacted


def test_redact_cli_flags():
    text = "--api-key=ABC123 --token DEF456 password=\"p@ss\""
    redacted = redact_text(text)
    assert "ABC123" not in redacted
    assert "DEF456" not in redacted
    assert "p@ss" not in redacted
    assert "--api-key" in redacted


def test_redact_multi_leak():
    text = "Authorization: Bearer SECRET token=XYZ url=http://x?api_key=HIDDEN"
    redacted = redact_text(text)
    assert "SECRET" not in redacted
    assert "XYZ" not in redacted
    assert "HIDDEN" not in redacted
    assert "Authorization" in redacted
