from __future__ import annotations

import json
import math
import unittest

from tangs.artifacts import canonical_json


class ArtifactTests(unittest.TestCase):
    def test_infinite_cap_is_strict_json(self):
        payload = canonical_json({"tau": math.inf})
        self.assertEqual(json.loads(payload), {"tau": "inf"})
        self.assertNotIn("Infinity", payload)


if __name__ == "__main__":
    unittest.main()
