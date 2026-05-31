import unittest
import tempfile
import os
import gzip
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parsers import load_matrix_csc_file, load_matrix_mtx_gz, normalize_file_path, parse_csc_content
from upload_store import save_uploaded_matrix_file


class ParserTests(unittest.TestCase):
    def test_parse_csc_content(self):
        result = parse_csc_content("2 3\n0 1 2 3\n0 1 0\n1.0 3.0 2.0\n")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["A_rows"], [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    def test_load_csc_file_resolves_from_package_dir(self):
        result = load_matrix_csc_file("matrix_csc_example.txt")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["A_rows"], [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    def test_normalize_file_path_extracts_filename_from_natural_language(self):
        path = normalize_file_path("matrix_csc_example.txt 文件中的矩阵")

        self.assertEqual(path, "matrix_csc_example.txt")

    def test_load_csc_file_tolerates_natural_language_suffix(self):
        result = load_matrix_csc_file("matrix_csc_example.txt 文件中的矩阵")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["A_rows"], [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        self.assertIn("resolved_path", result)

    def test_load_csc_file_resolves_uploaded_file_id(self):
        old_upload_dir = os.environ.get("NLA_UPLOAD_DIR")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["NLA_UPLOAD_DIR"] = tmpdir
                metadata = save_uploaded_matrix_file(
                    "uploaded.csc",
                    b"2 3\n0 1 2 3\n0 1 0\n1.0 3.0 2.0\n",
                    "text/plain",
                )

                result = load_matrix_csc_file(metadata["uri"])
        finally:
            if old_upload_dir is None:
                os.environ.pop("NLA_UPLOAD_DIR", None)
            else:
                os.environ["NLA_UPLOAD_DIR"] = old_upload_dir

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["shape"], [2, 3])
        self.assertEqual(result["A_rows"], [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])

    def test_load_mtx_gz_resolves_uploaded_uri(self):
        old_upload_dir = os.environ.get("NLA_UPLOAD_DIR")
        mtx = b"%%MatrixMarket matrix coordinate real general\n2 2 2\n1 1 4.0\n2 2 5.0\n"
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                os.environ["NLA_UPLOAD_DIR"] = tmpdir
                metadata = save_uploaded_matrix_file("uploaded.mtx.gz", gzip.compress(mtx), "application/gzip")

                result = load_matrix_mtx_gz(metadata["uri"])
        finally:
            if old_upload_dir is None:
                os.environ.pop("NLA_UPLOAD_DIR", None)
            else:
                os.environ["NLA_UPLOAD_DIR"] = old_upload_dir

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["shape"], [2, 2])
        self.assertEqual(result["A_csc"]["nnz"], 2)

    def test_browser_path_error_is_protocol_guided(self):
        result = load_matrix_mtx_gz("Current Folder/494_bus.mtx.gz")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "file_resolution_error")
        self.assertEqual(result["detail"]["reference_type"], "browser_path")
        self.assertIn("next_allowed_actions", result)
        self.assertNotIn("checked_paths", result)

    def test_bare_filename_error_does_not_suggest_checked_paths(self):
        result = load_matrix_mtx_gz("494_bus.mtx.gz")

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error_type"], "file_resolution_error")
        self.assertEqual(result["detail"]["reference_type"], "bare_filename")
        self.assertNotIn("checked_paths", result)


if __name__ == "__main__":
    unittest.main()
