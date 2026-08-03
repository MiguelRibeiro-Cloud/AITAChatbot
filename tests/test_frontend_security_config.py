import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class FrontendSecurityConfigTests(unittest.TestCase):
    def test_static_web_app_sets_security_headers_and_blocks_source_maps(self):
        config = json.loads((ROOT / "frontend" / "public" / "staticwebapp.config.json").read_text())

        headers = config["globalHeaders"]
        self.assertIn("Content-Security-Policy", headers)
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        self.assertIn("connect-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cross-Origin-Opener-Policy"], "same-origin")
        self.assertIn("Permissions-Policy", headers)
        self.assertIn({"route": "/static/*.map", "statusCode": 404}, config["routes"])

    def test_build_disables_source_map_generation(self):
        package_json = json.loads((ROOT / "frontend" / "package.json").read_text())

        self.assertIn("GENERATE_SOURCEMAP=false", package_json["scripts"]["build"])

    def test_ai_markdown_rendering_is_restricted(self):
        app_source = (ROOT / "frontend" / "src" / "App.js").read_text()

        self.assertIn("MARKDOWN_ALLOWED_ELEMENTS", app_source)
        self.assertIn("allowedElements={MARKDOWN_ALLOWED_ELEMENTS}", app_source)
        self.assertIn("unwrapDisallowed", app_source)
        self.assertNotIn("'a'", app_source.partition("const MARKDOWN_ALLOWED_ELEMENTS")[2].split(";", 1)[0])
        self.assertNotIn("'img'", app_source.partition("const MARKDOWN_ALLOWED_ELEMENTS")[2].split(";", 1)[0])

    def test_privacy_and_terms_disclosures_are_linked(self):
        app_source = (ROOT / "frontend" / "src" / "App.js").read_text()

        self.assertIn("Privacy Policy", app_source)
        self.assertIn("Terms", app_source)
        self.assertIn("Do not include identifying details", app_source)
        self.assertIn("Google GenAI", app_source)
        self.assertIn("Azure Functions API", app_source)


if __name__ == "__main__":
    unittest.main()
