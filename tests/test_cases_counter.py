import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = ROOT / "api"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeHttpResponse:
    def __init__(self, body=None, status_code=200, mimetype=None, headers=None):
        self.body = body
        self.status_code = status_code
        self.mimetype = mimetype
        self.headers = headers or {}

    def get_body(self):
        if isinstance(self.body, bytes):
            return self.body
        return (self.body or "").encode()


class FakeRequest:
    def __init__(self, payload, headers=None):
        self.payload = payload
        self.headers = headers or {}

    def get_json(self):
        return self.payload


def install_fake_azure_functions():
    azure = types.ModuleType("azure")
    functions = types.ModuleType("azure.functions")
    functions.HttpResponse = FakeHttpResponse
    functions.HttpRequest = FakeRequest
    azure.functions = functions
    sys.modules["azure"] = azure
    sys.modules["azure.functions"] = functions


class DbCounterTests(unittest.TestCase):
    def load_db_with_fake_psycopg(self, rows):
        executed = []
        connections = []

        class FakeCursor:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def execute(self, sql):
                executed.append(sql)

            def fetchone(self):
                return rows.pop(0)

        class FakeConnection:
            def __init__(self, kwargs):
                self.kwargs = kwargs
                self.exited = False

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.exited = True
                return False

            def cursor(self):
                return FakeCursor()

        def fake_connect(**kwargs):
            conn = FakeConnection(kwargs)
            connections.append(conn)
            return conn

        sys.modules["psycopg"] = types.SimpleNamespace(connect=fake_connect)
        db = load_module("db_under_test", API_ROOT / "db.py")
        return db, executed, connections

    def test_read_count_calls_approved_function(self):
        db, executed, _connections = self.load_db_with_fake_psycopg(rows=[(123,)])
        env = {
            "POSTGRES_HOST": "example.postgres.database.azure.com",
            "POSTGRES_PORT": "5432",
            "POSTGRES_DATABASE": "aitabot",
            "POSTGRES_USER": "aitabot_app",
            "POSTGRES_PASSWORD": "secret",
            "POSTGRES_SSLMODE": "require",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(db.get_cases_heard(), 123)

        self.assertEqual(executed, ["SELECT counter.get_cases_heard();"])

    def test_increment_count_calls_approved_atomic_function(self):
        db, executed, connections = self.load_db_with_fake_psycopg(rows=[(124,)])
        env = {
            "POSTGRES_HOST": "example.postgres.database.azure.com",
            "POSTGRES_DATABASE": "aitabot",
            "POSTGRES_USER": "aitabot_app",
            "POSTGRES_PASSWORD": "secret",
        }

        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(db.increment_cases_heard(), 124)

        self.assertEqual(executed, ["SELECT counter.increment_cases_heard();"])
        self.assertTrue(connections[0].exited)

    def test_db_helper_does_not_directly_modify_counter_table(self):
        source = (API_ROOT / "db.py").read_text()

        self.assertNotIn("UPDATE counter.cases_heard", source)
        self.assertNotIn("INSERT INTO counter.cases_heard", source)
        self.assertNotIn("DELETE FROM counter.cases_heard", source)

    def test_missing_environment_variables_raise_controlled_failure(self):
        db, _executed, _connections = self.load_db_with_fake_psycopg(rows=[(1,)])

        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(db.CounterConfigError):
                db.get_cases_heard()


class ApiCounterIntegrationTests(unittest.TestCase):
    def setUp(self):
        install_fake_azure_functions()

    def install_fake_shared_code(self, *, model_raises=False, schema_fails=False, verdict="guilty"):
        provider_error = "provider failed with password=super-secret at /tmp/private/path"

        shared_code = types.ModuleType("shared_code")
        class ProviderBusyError(RuntimeError):
            pass

        class StructuredResponseError(ValueError):
            pass

        class RequestValidationError(ValueError):
            pass

        def generate_judgment(_contents):
            if model_raises:
                raise RuntimeError(provider_error)
            if schema_fails:
                raise StructuredResponseError("Structured response was invalid JSON.")
            verdict_line = (
                "The Court Declares: Guilty!" if verdict == "guilty"
                else "The Court Declares: Not Guilty!"
            )
            return {
                "verdict": verdict,
                "reply": f"{verdict_line}\n\nCase complete.\n\nThe bailiff polishes a spoon.",
            }

        shared_code.ProviderBusyError = ProviderBusyError
        shared_code.ProviderTimeoutError = TimeoutError
        shared_code.RequestValidationError = RequestValidationError
        shared_code.StructuredResponseError = StructuredResponseError
        shared_code.build_contents = lambda history, user_message: [{"role": "user", "parts": [{"text": user_message}]}]
        shared_code.check_rate_limit = lambda req: (True, None)
        shared_code.classify_genai_error = lambda exc: ("unknown", str(exc))
        shared_code.generate_judgment = mock.Mock(side_effect=generate_judgment)
        shared_code.run_with_timeout = lambda fn: fn()
        shared_code.validate_chat_payload = lambda data: (data["message"].strip(), data.get("history", []))
        sys.modules["shared_code"] = shared_code
        return shared_code

    def install_fake_db(self, *, increment_value=125, increment_raises=False):
        calls = {"get": 0, "increment": 0}
        db = types.ModuleType("db")

        class CounterConfigError(RuntimeError):
            pass

        class CounterDatabaseError(RuntimeError):
            pass

        def get_cases_heard():
            calls["get"] += 1
            return 123

        def increment_cases_heard():
            calls["increment"] += 1
            if increment_raises:
                raise CounterDatabaseError("counter unavailable")
            return increment_value

        db.CounterConfigError = CounterConfigError
        db.CounterDatabaseError = CounterDatabaseError
        db.get_cases_heard = get_cases_heard
        db.increment_cases_heard = increment_cases_heard
        sys.modules["db"] = db
        return calls

    def test_clear_chat_does_not_reset_deployment_wide_count(self):
        source = (ROOT / "frontend" / "src" / "App.js").read_text()
        clear_chat_start = source.index("const clearChat = () => {")
        clear_chat_end = source.index("  const exportChat = () => {", clear_chat_start)
        clear_chat_source = source[clear_chat_start:clear_chat_end]

        self.assertNotIn("setMessageCount", clear_chat_source)

    def test_chat_response_serializes_cases_heard_as_json_number(self):
        self.install_fake_shared_code()
        calls = self.install_fake_db(increment_value=9007199254740993)
        chat = load_module("chat_function_under_test", API_ROOT / "chat" / "__init__.py")

        response = chat.main(FakeRequest({"message": "AITA?", "history": []}))
        payload = json.loads(response.get_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["casesHeard"], 9007199254740993)
        self.assertIsInstance(payload["casesHeard"], int)
        self.assertEqual(set(payload), {"reply", "verdict", "casesHeard"})
        self.assertEqual(payload["verdict"], "guilty")
        self.assertEqual(
            payload["reply"],
            "The Court Declares: Guilty!\n\nCase complete.\n\nThe bailiff polishes a spoon.",
        )
        self.assertEqual(calls["increment"], 1)

    def test_stream_success_increments_exactly_once(self):
        self.install_fake_shared_code()
        calls = self.install_fake_db(increment_value=126)
        stream = load_module("stream_function_under_test", API_ROOT / "chat_stream" / "__init__.py")

        response = stream.main(FakeRequest({"message": "AITA?", "history": []}))
        body = response.get_body().decode()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(calls["increment"], 1)
        self.assertIn('"casesHeard": 126', body)
        self.assertIn('"verdict": "guilty"', body)
        self.assertNotIn('"debug"', body)
        self.assertNotIn('"model"', body)
        self.assertNotIn("first_chunk_shapes", body)

    def test_both_chat_paths_use_shared_structured_judgment(self):
        shared_code = self.install_fake_shared_code(verdict="not_guilty")
        calls = self.install_fake_db()
        chat = load_module("structured_chat_function_under_test", API_ROOT / "chat" / "__init__.py")
        stream = load_module("structured_stream_function_under_test", API_ROOT / "chat_stream" / "__init__.py")

        chat_response = chat.main(FakeRequest({"message": "AITA?"}))
        stream_response = stream.main(FakeRequest({"message": "AITA?"}))
        chat_payload = json.loads(chat_response.get_body())
        events = [json.loads(line[6:]) for line in stream_response.get_body().decode().splitlines() if line.startswith("data: ")]

        self.assertEqual(shared_code.generate_judgment.call_count, 2)
        self.assertEqual(chat_payload["verdict"], "not_guilty")
        self.assertEqual(chat_payload["reply"], "The Court Declares: Not Guilty!\n\nCase complete.\n\nThe bailiff polishes a spoon.")
        self.assertEqual(events[0], {"token": chat_payload["reply"]})
        self.assertEqual(events[1]["verdict"], chat_payload["verdict"])
        self.assertEqual(calls["increment"], 2)

    def test_schema_failure_is_safe_on_both_chat_paths(self):
        self.install_fake_shared_code(schema_fails=True)
        calls = self.install_fake_db()
        chat = load_module("schema_error_chat_function_under_test", API_ROOT / "chat" / "__init__.py")
        stream = load_module("schema_error_stream_function_under_test", API_ROOT / "chat_stream" / "__init__.py")

        with mock.patch.object(chat.logging, "exception"), mock.patch.object(stream.logging, "exception"):
            chat_response = chat.main(FakeRequest({"message": "AITA?"}))
            stream_response = stream.main(FakeRequest({"message": "AITA?"}))

        self.assertEqual(chat_response.status_code, 502)
        self.assertEqual(json.loads(chat_response.get_body()), {"error": "The request could not be completed."})
        self.assertEqual(stream_response.status_code, 502)
        self.assertNotIn("verdict", stream_response.get_body().decode())
        self.assertNotIn("invalid JSON", stream_response.get_body().decode())
        self.assertEqual(calls["increment"], 0)

    def test_failed_model_request_does_not_increment(self):
        self.install_fake_shared_code(model_raises=True)
        calls = self.install_fake_db()
        stream = load_module("failed_stream_function_under_test", API_ROOT / "chat_stream" / "__init__.py")

        with mock.patch.object(stream.logging, "exception") as log_exception:
            response = stream.main(FakeRequest({"message": "AITA?", "history": []}))

        self.assertNotEqual(response.status_code, 200)
        self.assertEqual(calls["increment"], 0)
        log_exception.assert_called_once()

    def test_json_chat_errors_do_not_return_exception_message_and_are_logged(self):
        self.install_fake_shared_code(model_raises=True)
        calls = self.install_fake_db()
        chat = load_module("failed_chat_function_under_test", API_ROOT / "chat" / "__init__.py")

        with mock.patch.object(chat.logging, "exception") as log_exception:
            response = chat.main(FakeRequest({"message": "AITA?", "history": []}))

        body = response.get_body().decode()
        payload = json.loads(body)

        self.assertEqual(response.status_code, 500)
        self.assertEqual(payload["error"], "The request could not be completed.")
        self.assertEqual(set(payload), {"error"})
        self.assertNotIn("provider failed", body)
        self.assertNotIn("super-secret", body)
        self.assertNotIn("/tmp/private/path", body)
        self.assertEqual(calls["increment"], 0)
        log_exception.assert_called_once()

    def test_streaming_chat_errors_do_not_return_exception_message_and_are_logged(self):
        self.install_fake_shared_code(model_raises=True)
        calls = self.install_fake_db()
        stream = load_module("sanitized_stream_function_under_test", API_ROOT / "chat_stream" / "__init__.py")

        with mock.patch.object(stream.logging, "exception") as log_exception:
            response = stream.main(FakeRequest({"message": "AITA?", "history": []}))

        body = response.get_body().decode()

        self.assertEqual(response.status_code, 500)
        self.assertIn('"error": "The request could not be completed."', body)
        self.assertNotIn('"debug"', body)
        self.assertNotIn('"model"', body)
        self.assertNotIn("classified_kind", body)
        self.assertNotIn("provider failed", body)
        self.assertNotIn("super-secret", body)
        self.assertNotIn("/tmp/private/path", body)
        self.assertEqual(calls["increment"], 0)
        log_exception.assert_called_once()

    def test_counter_read_endpoint_does_not_increment(self):
        calls = self.install_fake_db()
        cases_heard = load_module("cases_heard_function_under_test", API_ROOT / "cases_heard" / "__init__.py")

        response = cases_heard.main(FakeRequest({}))
        payload = json.loads(response.get_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload, {"casesHeard": 123})
        self.assertEqual(calls, {"get": 1, "increment": 0})

    def test_database_failure_returns_safe_counter_error(self):
        calls = self.install_fake_db(increment_raises=True)
        sys.modules["db"].get_cases_heard = mock.Mock(side_effect=sys.modules["db"].CounterDatabaseError("secret internals"))
        cases_heard = load_module("cases_heard_error_function_under_test", API_ROOT / "cases_heard" / "__init__.py")

        response = cases_heard.main(FakeRequest({}))
        payload = json.loads(response.get_body())

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload, {"error": "Cases-heard counter is temporarily unavailable."})
        self.assertEqual(calls["increment"], 0)
        self.assertNotIn("secret", response.get_body().decode())

    def test_health_endpoint_returns_status_only(self):
        health = load_module("health_function_under_test", API_ROOT / "health" / "__init__.py")

        response = health.main(FakeRequest({}))
        payload = json.loads(response.get_body())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload, {"status": "alive"})


if __name__ == "__main__":
    unittest.main()
