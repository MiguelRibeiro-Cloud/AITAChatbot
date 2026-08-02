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
    def __init__(self, payload):
        self.payload = payload

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

    def install_fake_shared_code(self, *, model_raises=False):
        provider_error = "provider failed with password=super-secret at /tmp/private/path"

        class FakeResponse:
            text = "The Court Declares: Guilty!\n\nCase complete."

        class FakeModels:
            def generate_content(self, **_kwargs):
                if model_raises:
                    raise RuntimeError(provider_error)
                return FakeResponse()

            def generate_content_stream(self, **_kwargs):
                if model_raises:
                    raise RuntimeError(provider_error)
                return [types.SimpleNamespace(text="The Court Declares: Guilty!\n\nCase complete.")]

        shared_code = types.ModuleType("shared_code")
        shared_code.COCONUT_FALLBACK = "fallback"
        shared_code.MAX_OUTPUT_TOKENS = 1024
        shared_code.MODEL_NAME = "test-model"
        shared_code.SYSTEM_INSTRUCTION = "system"
        shared_code.build_contents = lambda history, user_message: [{"role": "user", "parts": [{"text": user_message}]}]
        shared_code.classify_genai_error = lambda exc: ("unknown", str(exc))
        shared_code.clean_reply = lambda text: text
        shared_code.client = types.SimpleNamespace(models=FakeModels())
        shared_code.extract_reply_text = lambda response: (response.text, None)
        shared_code.response_diagnostics = lambda response: {}
        shared_code.user_facing_error_message = lambda exc: "safe public error"
        sys.modules["shared_code"] = shared_code

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
        self.assertEqual(payload["debug"]["stage"], "genai_call")
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
        self.assertIn('"message": "The request could not be completed."', body)
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


if __name__ == "__main__":
    unittest.main()
