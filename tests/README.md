# Podcast Engine - Test Suite

Automated test suite for the Podcast Engine microservice.

## 📊 Test Coverage

| Module | Tests | Coverage Target |
|--------|-------|----------------|
| **test_chunking.py** | 13 tests | Text chunking, preprocessing, sentence splitting |
| **test_api.py** | 11 tests | FastAPI endpoints, health checks, voices API |
| **test_tts.py** | 7 tests | TTS client, parallel synthesis, retry logic |
| **Total** | **31 tests** | **Target: 70%+ coverage** |

---

## 🚀 Running Tests

### Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest

# Run specific test file
pytest tests/test_chunking.py

# Run with coverage report
pytest --cov=app --cov-report=html

# Run only fast tests
pytest -m unit

# Run with verbose output
pytest -v -s
```

### Docker (Production-like environment)

```bash
# Build test image
docker build -t podcast-engine:test --target test .

# Run tests in container
docker run --rm podcast-engine:test pytest

# Run with coverage
docker run --rm podcast-engine:test pytest --cov=app --cov-report=term-missing
```

---

## 📝 Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest fixtures and mocks
├── test_chunking.py         # Text chunking tests
├── test_api.py              # API endpoint tests
├── test_tts.py              # TTS client tests
└── README.md                # This file
```

### Key Fixtures (conftest.py)

- **`mock_kokoro_tts`**: Mocks Kokoro TTS API calls
- **`mock_ffmpeg`**: Mocks ffmpeg audio processing
- **`temp_storage`**: Creates temporary test directories
- **`sample_text_short/long`**: Sample text for testing
- **`sample_podcast_request`**: Valid podcast creation request

---

## 🧪 Test Categories

### Unit Tests (`test_chunking.py`, `test_tts.py`)

Fast tests with no external dependencies (< 100ms each).

```bash
pytest -m unit
```

### API Tests (`test_api.py`)

Integration tests for FastAPI endpoints.

```bash
pytest -m api
```

### Async Tests

All TTS and API tests use `pytest-asyncio` for async/await support.

```bash
pytest --asyncio-mode=auto
```

---

## 📈 Coverage Reports

### Generate HTML Report

```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html  # View in browser
```

### Coverage Requirements

- **Minimum**: 70% coverage (enforced)
- **Target**: 80%+ coverage
- **Excluded**: `__init__.py`, test files, venv

Configuration in `.coveragerc` and `pytest.ini`.

---

## 🎯 Test Scenarios Covered

### Text Chunking (`test_chunking.py`)

- ✅ Basic chunking with size limits
- ✅ Sentence-aware splitting
- ✅ Markdown removal (`**bold**`, `*italic*`)
- ✅ URL removal
- ✅ Chapter detection (H1-H6)
- ✅ Whitespace normalization
- ✅ Long sentence handling
- ✅ Empty text handling

### API Endpoints (`test_api.py`)

- ✅ Root endpoint (service info)
- ✅ Health check (Kokoro TTS status)
- ✅ Voices API (all voices, language filtering)
- ✅ Podcast creation (success, validation)
- ✅ Speed validation (0.5-2.0 range)
- ✅ GUI homepage (HTML, Alpine.js)

### TTS Client (`test_tts.py`)

- ✅ Basic synthesis
- ✅ Parallel synthesis with semaphore
- ✅ Retry logic on timeout
- ✅ Chapter markers prepending
- ✅ Available voices fetching
- ✅ Health check

---

## 🐛 Debugging Failed Tests

### View Detailed Output

```bash
pytest -vv --showlocals
```

### Run Single Test

```bash
pytest tests/test_chunking.py::TestTextChunker::test_basic_chunking -v
```

### Drop into Debugger on Failure

```bash
pytest --pdb
```

### Check Logs

Tests use `loguru` for logging. Logs are visible with `-s` flag:

```bash
pytest -s
```

---

## 🔧 Writing New Tests

### Test Template

```python
import pytest

class TestNewFeature:
    \"\"\"Tests for new feature\"\"\"

    def test_basic_functionality(self, mock_kokoro_tts, temp_storage):
        \"\"\"Test basic feature works\"\"\"
        # Arrange
        input_data = "test input"

        # Act
        result = my_function(input_data)

        # Assert
        assert result == expected_output

    @pytest.mark.asyncio
    async def test_async_feature(self, async_client):
        \"\"\"Test async endpoint\"\"\"
        response = await async_client.get("/endpoint")
        assert response.status_code == 200
```

### Using Fixtures

```python
def test_with_mocks(self, mock_kokoro_tts, mock_ffmpeg, temp_storage):
    # mock_kokoro_tts: TTS calls return fake audio
    # mock_ffmpeg: ffmpeg returns fake merged audio
    # temp_storage: Temporary directories auto-cleaned
    pass
```

---

## 🚨 CI/CD Integration

Tests run automatically on:
- ✅ Pull requests
- ✅ Push to `main` branch
- ✅ Manual workflow dispatch

See `.github/workflows/test.yml` for CI configuration.

---

## 📚 Dependencies

Test dependencies (from `requirements.txt`):

```txt
pytest==8.3.3                # Test framework
pytest-asyncio==0.24.0       # Async test support
pytest-cov==6.0.0            # Coverage plugin
coverage[toml]==7.6.10       # Coverage measurement
httpx==0.27.2                # Async HTTP client (also runtime)
```

---

## 🎓 Best Practices

1. **Mock External Services**: Always mock Kokoro TTS and ffmpeg
2. **Use Fixtures**: Reuse common setup via fixtures
3. **Test Edge Cases**: Empty text, invalid inputs, timeouts
4. **Keep Tests Fast**: Unit tests < 100ms, integration tests < 1s
5. **Clean Up**: Use `tmp_path` for temporary files (auto-cleaned)
6. **Descriptive Names**: `test_sentence_aware_splitting` not `test_1`
7. **One Assert Per Test**: Focus on single behavior

---

## 🆘 Troubleshooting

### Import Errors

```bash
# Ensure app/ is in PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
pytest
```

### Async Warnings

```bash
# Use auto mode
pytest --asyncio-mode=auto
```

### Coverage Not Working

```bash
# Install pytest-cov
pip install pytest-cov coverage
```

---

**Last Updated**: 2025-10-12
**Test Suite Version**: 1.0.0
**Total Tests**: 31
**Coverage Target**: 70%+
