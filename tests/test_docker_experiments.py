import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

from benchmarks.benchmark_utils import run_rest_benchmark, create_dummy_jpeg


@pytest.mark.asyncio
async def test_run_rest_benchmark_structure():
    #verify run_rest_benchmark returns correct JSON structure
    with patch('aiohttp.ClientSession.post') as mock_post:
        mock_response = AsyncMock()
        mock_response.json = AsyncMock(return_value={"class_id": 0})
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        with patch('aiohttp.ClientSession') as mock_session_class:
            mock_session = AsyncMock()
            mock_session.post.return_value = mock_response
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            mock_session_class.return_value = mock_session

            result = await run_rest_benchmark(
                "http://test.local/predict",
                concurrency=1,
                num_requests=5
            )

            #verify result has required keys
            assert "concurrency" in result
            assert "num_requests" in result
            assert "rps" in result
            assert "avg_latency_ms" in result
            assert "p50" in result
            assert "p95" in result
            assert "p99" in result
            assert "failed" in result


@pytest.mark.asyncio
async def test_run_rest_benchmark_empty_latencies():
    #when all requests fail, still return valid structure with zeros
    with patch('aiohttp.ClientSession') as mock_session_class:
        mock_session = AsyncMock()
        mock_session.post.side_effect = Exception("Connection failed")
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        mock_session_class.return_value = mock_session

        result = await run_rest_benchmark(
            "http://test.local/predict",
            concurrency=1,
            num_requests=5
        )

        #should handle all failures gracefully
        assert result["rps"] == 0
        assert result["avg_latency_ms"] == 0
        assert result["failed"] == 5


def test_create_dummy_jpeg_size():
    #verify dummy JPEG is created with correct size
    jpeg_data = create_dummy_jpeg(size=(32, 32))
    assert isinstance(jpeg_data, bytes)
    assert len(jpeg_data) > 0

    #verify it's a valid JPEG
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(jpeg_data))
    assert img.size == (32, 32)
    assert img.format == "JPEG"


def test_results_directory_exists():
    #verify results directory is created
    assert Path("benchmarks/results").exists()
    assert Path("benchmarks/results").is_dir()
