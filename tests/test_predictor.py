import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock


class TestPredictorDevice:

    def _make_mock_jit_model(self):
        mock = MagicMock()
        mock.to.return_value = mock
        mock.eval.return_value = mock
        return mock

    def _make_mock_nn_model(self):
        mock = MagicMock()
        mock.to.return_value = mock
        mock.eval.return_value = mock
        return mock

    @patch("src.inference.predictor.torch.jit.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_predictor_defaults_to_cpu(self, mock_exists, mock_jit_load):
        mock_jit_load.return_value = self._make_mock_jit_model()
        from src.inference.predictor import Predictor
        p = Predictor("model.pt")
        assert p.device == torch.device("cpu")

    @patch("src.inference.predictor.torch.jit.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_predictor_accepts_device_cpu(self, mock_exists, mock_jit_load):
        mock_jit_load.return_value = self._make_mock_jit_model()
        from src.inference.predictor import Predictor
        p = Predictor("model.pt", device="cpu")
        assert p.device == torch.device("cpu")

    @patch("src.inference.predictor.torch.jit.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_predictor_accepts_device_mps(self, mock_exists, mock_jit_load):
        mock_jit_load.return_value = self._make_mock_jit_model()
        from src.inference.predictor import Predictor
        p = Predictor("model.pt", device="mps")
        assert p.device == torch.device("mps")

    @patch("src.inference.predictor.torch.jit.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_torchscript_model_moved_to_device(self, mock_exists, mock_jit_load):
        mock_model = self._make_mock_jit_model()
        mock_jit_load.return_value = mock_model
        from src.inference.predictor import Predictor
        p = Predictor("model.pt", device="cpu")
        mock_model.to.assert_called_with(torch.device("cpu"))

    @patch("src.inference.predictor.get_model")
    @patch("src.inference.predictor.torch.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_pytorch_model_moved_to_device(self, mock_exists, mock_torch_load, mock_get_model):
        mock_model = self._make_mock_nn_model()
        mock_get_model.return_value = mock_model
        mock_torch_load.return_value = {}
        from src.inference.predictor import Predictor
        p = Predictor("model.pth", device="cpu")
        mock_model.to.assert_called_with(torch.device("cpu"))

    @patch("src.inference.predictor.ort.InferenceSession")
    @patch("pathlib.Path.exists", return_value=True)
    def test_onnx_ignores_mps_device(self, mock_exists, mock_session):
        mock_session.return_value = MagicMock()
        from src.inference.predictor import Predictor
        p = Predictor("model.onnx", device="mps")
        assert p.device == torch.device("cpu")

    @patch("src.inference.predictor.get_model")
    @patch("src.inference.predictor.torch.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_predict_moves_tensor_to_device(self, mock_exists, mock_torch_load, mock_get_model):
        mock_model = self._make_mock_nn_model()
        logits = torch.zeros(1, 10)
        logits[0, 3] = 5.0
        mock_model.return_value = logits
        mock_get_model.return_value = mock_model
        mock_torch_load.return_value = {}
        from src.inference.predictor import Predictor
        p = Predictor("model.pth", device="cpu")
        result = p.predict(torch.randn(1, 3, 32, 32))
        assert isinstance(result, int)

    @patch("src.inference.predictor.ort.InferenceSession")
    @patch("pathlib.Path.exists", return_value=True)
    def test_onnx_predict_does_not_crash_with_cpu_tensor(self, mock_exists, mock_session_cls):
        mock_session = MagicMock()
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.run.return_value = [np.zeros((1, 10), dtype=np.float32)]
        mock_session_cls.return_value = mock_session
        from src.inference.predictor import Predictor
        p = Predictor("model.onnx", device="cpu")
        result = p.predict(torch.randn(1, 3, 32, 32))
        assert isinstance(result, int)
