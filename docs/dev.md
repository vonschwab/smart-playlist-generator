# Development Guide

Setup, testing, and debugging guide for Playlist Generator development.

## Prerequisites

- **Python 3.10+**: `python --version`
- **Node.js 18+**: `node --version` (for UI development)
- **Git**: `git --version`
- **ffmpeg** (optional): Better audio analysis

## Development Setup

### 1. Clone Repository

```bash
git clone https://github.com/vonschwab/smart-playlist-generator.git
cd smart-playlist-generator
```

### 2. Create Virtual Environment

**Windows**:
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux**:
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Install development extras:

```bash
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 mypy  # Testing + linting
```

### 4. Setup Configuration

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your music library path
```

### 5. Create Test Data (Optional)

For testing without full library:

```bash
python scripts/scan_library.py --limit 100  # Scan only first 100 tracks
```

## Project Structure

```
src/                          # Core library
├── playlist_generator.py      # Main playlist generation logic
├── similarity_calculator.py   # Scoring system
├── local_library_client.py    # Database queries
├── metadata_client.py         # Database management
├── librosa_analyzer.py        # Audio feature extraction
├── hybrid_sonic_analyzer.py   # Multi-segment analysis
├── config_loader.py           # Configuration management
└── ... (other modules)

scripts/                       # Operational scripts
├── scan_library.py            # Index music library
├── update_sonic.py            # Extract audio features
├── update_genres_v3_normalized.py  # Fetch genre data
└── analyze_library.py         # Unified pipeline

api/                           # FastAPI backend
├── main.py                    # REST API endpoints
├── services/                  # Business logic
└── models.py                  # Request/response models

ui/                            # React frontend
├── src/
├── public/
└── package.json

tests/                         # Test suite
├── unit/                      # Fast, no dependencies
├── integration/               # Database + fixtures
└── smoke/                     # CLI validation

docs/                          # Documentation
└── *.md                       # Markdown guides
```

## Running the Backend

### API Server

```bash
# Development (with auto-reload)
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Production
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Or using module:

```bash
python -m api
```

Access Swagger UI: **http://localhost:8000/docs**

### CLI Entrypoint

```bash
python main_app.py --artist "Artist Name" --count 50
```

Generate with specific mode:

```bash
python main_app.py --artist "Fela Kuti" --count 50 --dynamic
```

Dry run (preview without writing):

```bash
python main_app.py --artist "Fela Kuti" --count 50 --dry-run
```

## Running the Frontend

### Setup

```bash
cd ui
npm install
```

### Development Server

```bash
npm run dev
```

Access UI: **http://localhost:5173**

Hot-reload enabled for `.tsx` and `.css` files.

### Build for Production

```bash
npm run build
```

Output: `ui/dist/` (static files for deployment)

## Testing

### Run All Tests

```bash
pytest
```

With coverage report:

```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

### Run Specific Test Categories

```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests (requires database)
pytest tests/integration/

# Smoke tests (CLI validation)
pytest tests/smoke/

# Specific test file
pytest tests/unit/test_config_loader.py

# Specific test function
pytest tests/unit/test_config_loader.py::test_load_config
```

### Run Tests with Markers

```bash
# Fast tests only (skip slow)
pytest -m "not slow"

# Slow tests only
pytest -m slow

# Specific marker
pytest -m "audio"
```

### Test Configuration

Tests use `config.example.yaml` by default (no API keys required).

To test with real data:

```bash
cp config.yaml config.test.yaml
# Edit config.test.yaml as needed
pytest --config-file=config.test.yaml
```

## Debugging

### Enable Verbose Logging

Set logging level in config:

```yaml
logging:
  level: DEBUG
```

Or via environment variable:

```bash
export LOG_LEVEL=DEBUG
python main_app.py --artist "Fela Kuti" --count 10
```

### Debug a Function

Using Python debugger:

```python
# In your code:
import pdb; pdb.set_trace()

# Or use:
breakpoint()  # Python 3.7+
```

Run with debugger enabled:

```bash
python -m pdb main_app.py --artist "Fela Kuti" --count 10
```

### Profile Performance

```python
import cProfile
import pstats

cProfile.run('some_function()', 'profile_stats')
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumulative').print_stats(20)  # Top 20 functions
```

Or use built-in timing:

```python
import time

start = time.time()
result = expensive_operation()
elapsed = time.time() - start
print(f"Took {elapsed:.2f} seconds")
```

### Check Database Integrity

```bash
sqlite3 data/metadata.db

# Check table structure
.schema tracks

# Count tracks
SELECT COUNT(*) FROM tracks;

# Find tracks without features
SELECT COUNT(*) FROM tracks WHERE sonic_features IS NULL;
```

## Code Quality

### Run Linter

```bash
flake8 src/
```

Fix style issues:

```bash
black src/
```

### Type Checking

```bash
mypy src/
```

### Format Code

```bash
black src/ api/ scripts/
```

## Adding a New Feature

### Step 1: Create Feature Branch

```bash
git checkout -b feature/my-feature
```

### Step 2: Implement Feature

Add code in appropriate module:

- Core logic: `src/new_module.py`
- CLI: `main_app.py` or `scripts/`
- API: `api/main.py` or `api/services/`
- Tests: `tests/unit/` or `tests/integration/`

### Step 3: Write Tests

```python
# tests/unit/test_new_module.py
import pytest
from src.new_module import MyClass

def test_my_feature():
    obj = MyClass()
    result = obj.do_something()
    assert result == expected
```

Run tests:

```bash
pytest tests/unit/test_new_module.py -v
```

### Step 4: Document

- Update `docs/` if user-facing
- Add docstrings to functions
- Update `README.md` if major feature

### Step 5: Commit and Push

```bash
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
```

Create pull request on GitHub.

## Common Development Tasks

### Add a New Playlist Mode

1. Update `src/playlist/config.py` with new mode constraints
2. Implement scoring in `src/playlist/constructor.py`
3. Add CLI flag to `main_app.py`
4. Test with existing artifacts:
   ```bash
   python main_app.py --artist "Fela Kuti" --count 30 --my-new-mode
   ```

### Add a New Similarity Method

1. Implement in `src/genre_similarity_v2.py`:
   ```python
   def my_similarity_method(genres1, genres2) -> float:
       # Implementation
       return score
   ```
2. Add to `SIMILARITY_METHODS` dict
3. Update config schema to allow new method
4. Test:
   ```python
   from src.genre_similarity_v2 import GenreSimilarityV2
   gs = GenreSimilarityV2()
   score = gs.calculate_similarity(["rock"], ["pop"], method="my_method")
   ```

### Add a New Audio Feature

1. Extract in `src/librosa_analyzer.py`:
   ```python
   def extract_my_feature(y, sr):
       # Use librosa to compute
       return feature_values
   ```
2. Integrate into feature dict
3. Update artifact builder to include new feature
4. Rebuild artifacts:
   ```bash
   python scripts/analyze_library.py --force
   ```

### Debug Playlist Quality

1. Generate with logging:
   ```bash
   LOG_LEVEL=DEBUG python main_app.py --artist "Fela Kuti" --count 20
   ```
2. Check log file: `playlist_generator.log`
3. Look for:
   - Constraint violations
   - Low similarity scores
   - Rejected candidates
4. Modify constraints in config:
   ```yaml
   playlists:
     max_tracks_per_artist: 3
     min_genre_similarity: 0.25
   ```
5. Test again

## Database Maintenance

### Backup Database

```bash
cp data/metadata.db data/metadata.db.backup
```

### Reset Database

```bash
rm data/metadata.db
python scripts/scan_library.py
```

### Vacuum Database (optimize)

```bash
sqlite3 data/metadata.db VACUUM;
```

### Export Database Stats

```bash
python scripts/validate_metadata.py
```

## Performance Optimization

### Profile a Pipeline Stage

```bash
# Time full pipeline
time python scripts/analyze_library.py

# Profile specific stage
python -m cProfile -s cumulative scripts/update_sonic.py | head -50
```

### Memory Profiling

```bash
pip install memory-profiler

python -m memory_profiler scripts/update_sonic.py
```

### Parallel Processing

```bash
# Increase workers for audio analysis
python scripts/update_sonic.py --workers 8

# Monitor CPU usage
watch -n 1 'top -b -n 1 | head -20'
```

## Deployment

### Docker Setup

```dockerfile
FROM python:3.10
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t playlist-generator .
docker run -p 8000:8000 -v $(pwd)/data:/app/data playlist-generator
```

### Environment Variables

Supported overrides in production:

```bash
export LIBRARY_MUSIC_DIRECTORY=/path/to/music
export LASTFM_API_KEY=your_key
export LOG_LEVEL=INFO
```

## Troubleshooting Development Issues

### Import Errors

```bash
# Ensure venv is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Database Locked

```bash
# Close other processes
pkill -f "python.*playlist_generator"

# Or delete and rescan
rm data/metadata.db
python scripts/scan_library.py
```

### Audio Analysis Failures

```bash
# Install ffmpeg for better compatibility
# macOS:
brew install ffmpeg

# Linux:
sudo apt-get install ffmpeg

# Windows:
choco install ffmpeg
```

### Slow Tests

```bash
# Run only fast tests
pytest -m "not slow"

# Run with verbose output to see which tests are slow
pytest -v --durations=10
```

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "feat: description"`
4. Push to fork: `git push origin feature/my-feature`
5. Create pull request with:
   - Clear description of changes
   - Any new tests
   - Updated documentation

## Code Standards

- **Style**: Black (run `black src/`)
- **Linting**: Flake8 (run `flake8 src/`)
- **Type hints**: Added for new functions
- **Tests**: Required for new features
- **Documentation**: Docstrings for public APIs

## Getting Help

- Check existing [GitHub Issues](https://github.com/vonschwab/smart-playlist-generator/issues)
- Review test examples in `tests/`
- Check `docs/` for architectural details

