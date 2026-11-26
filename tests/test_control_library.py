"""Exercise the reference bundle through the same search/read contract used by chat."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from control_harness.library import ControlLibrary, LIBRARY, LibraryError, RESOURCE_FIELDS


class BundledRoutesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library = ControlLibrary()

    def test_specialists_and_ecosystems(self):
        cases = [
            ("用 CSV 做纯 NN 时序预测", "model", "controlkit/table.py"),
            ("train a thermal PINN with an independent test split", "model", "controlkit/thermal.py"),
            ("二阶对象分段保持做下一状态预测", "model", "cases/object-plant/train.py"),
            ("Python 核验预测矩阵", "controller", "controlkit/prediction.py"),
            ("在 MATLAB 里核验 MPC 的预测矩阵", "controller",
             ".agents/skills/verify-mpc-prediction/scripts/verify_increment_matrix.m"),
            ("用 Simulink 核对一阶 MPC", "controller", "cases/matlab-scenes/mpc1/MPC_QP_1st.m"),
            ("audit a Simulink to Python port", None, "controlkit/parity.py"),
            ("热管理专家，做 NMPC 模仿", "lightweight", "cases/heat-policy/train_policy.py"),
            ("behavior cloning with expert data", "lightweight", "controlkit/policy.py"),
        ]
        for query, role, code in cases:
            with self.subTest(query=query):
                result = self.library.search(query, role)
                self.assertEqual(result["matches"][0]["code"], code)
                self.assertTrue(result["matches"][0]["matched_terms"])

    def test_scenes_not_in_original_index(self):
        for query, name in (("MATLAB 调 PI 增益", "pi"), ("battery state of charge EKF", "soc"),
                            ("电池热阻双 EKF", "dekf"), ("VGT 电增压器采数", "eturbo"),
                            ("二阶 RC 在线辨识", "pulse"), ("状态空间 S-function", "sfunction"),
                            ("电池热参数遗传辨识", "thermal")):
            with self.subTest(query=query):
                result = self.library.search(query)
                self.assertEqual(result["matches"][0]["case"], f"cases/matlab-scenes/{name}/README.md")

    def test_conversational_requests_do_not_require_an_exact_index_phrase(self):
        examples = [
            ("用 Simulink 搭二阶对象做闭环 MPC，和 ADRC 比较", "cases/matlab-scenes/mpc2/README.md"),
            ("仅有温度和功率的时序 CSV，按整条工况训练并留出测试", "controlkit/table.py"),
            ("把耗时的 MPC 蒸馏成一个实时可运行的小网络", "clone-control-policy"),
            ("给电池热管理做控制器蒸馏，对比学生网络和专家", "distill-thermal-expert"),
            ("将现有 Simulink 动态系统转换到 Python 后，检查重置和阶跃轨迹", "controlkit/parity.py"),
        ]
        for query, expected in examples:
            with self.subTest(query=query):
                first = self.library.search(query)["matches"][0]
                self.assertIn(expected, (first["case"], first["code"], first["skill"]))

    def test_torchscript_archives_are_not_readable_text(self):
        for model in ("coupled", "imitation", "nominal"):
            path = f"runs/heat-policy/{model}/inference.ts"
            self.assertIn(path, self.library.files)
            with self.assertRaisesRegex(LibraryError, "Binary"):
                self.library.read(path)

    def test_each_indexed_reference_exists_and_is_readable(self):
        # Exercise actual packaged paths including current working-tree MATLAB cases.
        # Deduplicate shared files such as the sampling knowledge chapter.
        paths = {route[field] for route in self.library._routes() for field in RESOURCE_FIELDS}
        for path in paths:
            with self.subTest(path=path):
                self.assertTrue(self.library.read(path, length=1)["text"])

    def test_no_unrelated_or_cross_role_match(self):
        self.assertEqual(self.library.search("summarize a novel about a doctor treating a factory worker")["matches"], [])
        self.assertEqual(self.library.search("模型采数", role="lightweight")["matches"], [])
        with self.assertRaises(LibraryError):
            self.library.search("核验", limit=100)

    def test_cli_used_from_source_directory(self):
        args = [sys.executable, "-B", "-S", "-E", "-X", "utf8", "-m", "control_harness.library"]
        process = subprocess.run(args + ["--query=电池热阻双 EKF", "--limit=1"], check=True,
                                 cwd=LIBRARY.parents[1] / "src", capture_output=True)
        first = json.loads(process.stdout)["matches"][0]
        process = subprocess.run(args + ["--read=" + first["case"], "--length=1000"], check=True,
                                 cwd=LIBRARY.parents[1] / "src", capture_output=True)
        result = json.loads(process.stdout)
        self.assertIn("MATLAB", result["text"])
        self.assertIn(first["code"], [link["path"] for link in result["links"]])


class ReferenceReadTest(unittest.TestCase):
    def setUp(self):
        output = Path(__file__).resolve().parents[1] / "test-output" / "control-library"
        output.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=output)
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.entries = {
            "cases/plant/README.md": "# 温度🌡\n[源程序](source.py) [部署模型](inference.ts) [外部](../../../private.txt)",
            "cases/plant/source.py": "# 本地程序",
            "cases/plant/inference.ts": "binary",
            "evaluation/private.py": "private",
            "runs/paired/first.py": "private",
            "cases/plant/model.slx": "binary",
        }
        files = {}
        for name, content in self.entries.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            data = content.encode("utf-8")
            path.write_bytes(data)
            files[name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (self.root / "snapshot.json").write_text(json.dumps({"schema_version": 1, "files": files}), encoding="utf-8")
        self.library = ControlLibrary(self.root)

    def test_unicode_pagination_and_relative_links(self):
        name = "cases/plant/README.md"
        first = self.library.read(name, length=4)
        second = self.library.read(name, offset=first["next_offset"])
        self.assertEqual(first["text"] + second["text"], self.entries[name])
        self.assertIsNone(second["next_offset"])
        self.assertEqual(first["sha256"], second["sha256"])
        self.assertEqual(second["links"], [{"path": "cases/plant/source.py", "readable": True},
                                           {"path": "cases/plant/inference.ts", "readable": False}])
        with self.assertRaises(LibraryError):
            self.library.read(name, offset=second["total_chars"] + 1)

    def test_reads_are_limited_to_intact_public_text(self):
        for name in ("../private.md", "/absolute.md", "C:/private.md", "cases/../evaluation/private.py",
                     "evaluation/private.py", "runs/paired/first.py", "cases/plant/model.slx", "unknown.md"):
            with self.subTest(name=name), self.assertRaises(LibraryError):
                self.library.read(name)
        name = "cases/plant/source.py"
        file = self.root / name
        file.write_bytes(b"x" * file.stat().st_size)
        with self.assertRaisesRegex(LibraryError, "checksum"):
            self.library.read(name)
