# Debugging Session - 2025-10-13

## 📋 Context
**Project**: Podcast Engine - Phase 3 (Async workflow implementation)
**Issue**: Jobs hang indefinitely when processing real articles, containers crash with exit code 137 (SIGKILL)
**Session Duration**: ~3 hours intensive debugging
**Status**: **ISSUE NOT RESOLVED** - Root cause identified, solution pending

---

## 🔴 Critical Issue Description

### Symptom
When async jobs process real articles (3+ chunks, 3954+ characters):
1. ✅ Job starts successfully (Step 1: Chunking completes)
2. ✅ Step 2: TTS synthesis launches (3 chunks submitted in parallel)
3. ❌ **HTTP requests to Kokoro TTS NEVER sent** (no logs in Kokoro service)
4. ❌ No "Received bytes" logs (requests block before sending)
5. ⏳ Job hangs indefinitely
6. 💀 Docker sends SIGTERM → Container killed with exit 137
7. 🔄 Docker Swarm restarts container

### What Works
- ✅ Health checks (11 characters, single request) → Complete successfully
- ✅ Sync mode (pre-Phase 3) → Worked perfectly
- ✅ Step 1 (Chunking) → Always completes

### What Doesn't Work
- ❌ Async jobs with 3+ chunks in parallel
- ❌ Multiple articles (2+ concurrent jobs)
- ❌ Any real content processing in async mode

---

## 🔍 Investigation Timeline

### Attempt 1: Enhanced Error Logging
**Commit**: `359b997`
**Action**: Added `logger.exception()` with full traceback to async job function
**Result**: ❌ **No exceptions captured** - Issue is a hang/deadlock, not an exception
**Key Discovery**: The problem is NOT throwing an exception

### Attempt 2: Dedicated TTS Client Per Job
**Commit**: `ddbb6b4`
**Hypothesis**: Shared `httpx.AsyncClient` causing connection pool exhaustion with multiple concurrent jobs
**Action**: Created dedicated `KokoroTTSClient()` for each async job with isolated connection pool
**Code**:
```python
# OLD (suspected deadlock):
tts_client = app_state.tts_client  # Shared global client

# NEW (isolated):
tts_client = KokoroTTSClient()  # Dedicated client per job
try:
    tts_results = await tts_client.synthesize_chunks_parallel(...)
finally:
    await tts_client.close()  # Always cleanup
```
**Result**: ❌ **No change** - Jobs still hang identically
**Key Discovery**: The problem is NOT the shared httpx.AsyncClient

### Attempt 3: Debug Logging Around HTTP Requests
**Commit**: `270550e` (FAILED BUILD)
**Action**: Added detailed logs around `await self.client.post()` to identify exact blocking point
**Result**: ❌ **Build failed** - Could not deploy
**Status**: Abandoned for now

---

## 🧠 Current Understanding

### Root Cause Hypothesis
The problem appears to be **`asyncio.gather()` with parallel tasks inside FastAPI BackgroundTasks**.

**Evidence**:
1. ✅ Single requests (health checks) work perfectly
2. ❌ 3 parallel requests via `asyncio.gather()` block completely
3. ❌ Requests never reach `await client.post()` (no HTTP traffic to Kokoro)
4. ❌ No exception raised (silent deadlock)

**Blocking Location**: Somewhere between:
- `await tts_client.synthesize_chunks_parallel()` (line 414 in main.py)
- `await self.client.post()` (line 87-93 in tts.py)

The async tasks appear to be waiting for something (event loop? semaphore?) that never completes.

### Key Observations
1. **Kokoro TTS logs show only 2 requests in 5 minutes** (health checks), not the 3 real chunks
2. **No HTTP traffic** means the problem is BEFORE the network call
3. **Container logs show**: `INFO: Waiting for background tasks to complete. (CTRL+C to force quit)`
4. **Pattern is 100% reproducible** across all attempts

---

## 🎯 Next Steps (Priority Order)

### Immediate Test (Highest Priority)
**Test Hypothesis**: Disable parallelization completely
- Rollback to commit `ddbb6b4` (last successful build)
- Change `max_parallel=5` to `max_parallel=1` in webhook endpoint
- Test if sequential processing works
- **Expected outcome**: If this works, confirms issue is async parallelism in BackgroundTasks

### If Sequential Works
1. **Root cause confirmed**: FastAPI BackgroundTasks incompatible with `asyncio.gather()` parallelism
2. **Solution options**:
   - Option A: Use Celery/RQ for true async job processing
   - Option B: Use FastAPI's async event loop differently
   - Option C: Keep parallelism but use different concurrency pattern (asyncio.create_task instead of gather)
   - Option D: Process chunks sequentially (slower but functional)

### If Sequential Fails
1. **Deeper investigation needed**: Problem is not parallelization
2. **Check**:
   - Event loop compatibility with FastAPI BackgroundTasks
   - httpx client initialization in async context
   - Docker Swarm networking issues
   - Resource limits (memory, file descriptors)

---

## 📊 Test Matrix

| Test Scenario | Health Check (11 chars) | Real Article (3 chunks) | Multiple Articles | Status |
|--------------|------------------------|------------------------|-------------------|--------|
| Sync Mode (pre-Phase 3) | ✅ Works | ✅ Works | ✅ Works | Deprecated |
| Async + Shared Client | ✅ Works | ❌ Hangs | ❌ Hangs | Commit `359b997` |
| Async + Dedicated Client | ✅ Works | ❌ Hangs | ❌ Hangs | Commit `ddbb6b4` |
| Async + Sequential | ⏳ Not tested | ⏳ Not tested | ⏳ Not tested | **NEXT** |

---

## 🔧 Technical Details

### Environment
- **FastAPI**: Background tasks with `async def`
- **httpx**: AsyncClient with connection pooling
- **asyncio**: `asyncio.gather()` for parallel chunk processing
- **Docker**: Swarm mode, health checks every 30s
- **Kokoro TTS**: Separate service, responds in ~2-3s per chunk

### Critical Code Paths

**Webhook endpoint** (`main.py:675-683`):
```python
@router.post("/webhook/create-podcast")
async def webhook_create_podcast(podcast_req: PodcastRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    logger.info(f"[{job_id}] Async mode enabled - submitting job to background")
    background_tasks.add_task(_process_podcast_job_async, job_id, podcast_req)
    return {"success": True, "job_id": job_id, "mode": "async"}
```

**Async job function** (`main.py:379-427`):
```python
async def _process_podcast_job_async(job_id: str, podcast_req: PodcastRequest):
    # Step 1: Chunking (WORKS)
    chunks = chunker.create_chunks(...)

    # Step 2: TTS synthesis (HANGS HERE)
    tts_client = KokoroTTSClient()  # Dedicated client
    try:
        tts_results = await tts_client.synthesize_chunks_parallel(
            chunks=chunks,
            max_parallel=podcast_req.processing_options.max_parallel_tts  # Default: 5
        )
    finally:
        await tts_client.close()
```

**Parallel synthesis** (`tts.py:145-104`):
```python
async def synthesize_chunks_parallel(self, chunks, max_parallel=5):
    semaphore = asyncio.Semaphore(max_parallel)

    async def process_chunk(chunk_id, text, chapter):
        async with semaphore:
            audio_bytes = await self.synthesize_chunk(text, voice, speed)  # BLOCKS HERE
            output_path.write_bytes(audio_bytes)

    tasks = [process_chunk(chunk_id, text, chapter) for chunk_id, text, chapter in chunks]
    results = await asyncio.gather(*tasks, return_exceptions=False)  # NEVER RETURNS
```

---

## 📝 Git History

```
270550e (HEAD, origin/main, main) - debug: Add detailed logging around httpx client.post() [FAILED BUILD]
ddbb6b4 - fix(async): Create dedicated TTS client per job to prevent deadlocks [DEPLOYED, STILL HANGS]
359b997 - feat(async): Add comprehensive error handling to async job function [DEPLOYED, NO EXCEPTION]
```

---

## 🤔 Open Questions

1. **Why do health checks work but real chunks don't?**
   - Same code path (`synthesize_chunk`)
   - Same httpx client setup
   - Only difference: 1 request vs 3 parallel requests

2. **Why no exception is raised?**
   - `logger.exception()` captures nothing
   - No timeout exception (despite httpx.Timeout config)
   - No network error
   - Silent deadlock/hang

3. **Why does Docker kill the container?**
   - Background tasks never complete
   - FastAPI shutdown waits for tasks
   - Docker sends SIGTERM after timeout
   - Container killed with exit 137

4. **Is this a known FastAPI BackgroundTasks limitation?**
   - Need to research FastAPI + asyncio.gather compatibility
   - Check if BackgroundTasks use a different event loop
   - Validate if Celery/RQ is recommended for heavy async work

---

## 📚 Resources for Further Investigation

- FastAPI BackgroundTasks documentation: https://fastapi.tiangolo.com/tutorial/background-tasks/
- asyncio.gather() deadlock patterns
- httpx AsyncClient in FastAPI background tasks
- Docker Swarm container lifecycle and SIGTERM handling
- Celery vs FastAPI BackgroundTasks comparison

---

## ✅ What We Know For Sure

1. ✅ Sync mode worked perfectly (pre-Phase 3)
2. ✅ Health checks work in async mode
3. ✅ Chunking always completes
4. ✅ The issue is specifically in parallel TTS synthesis
5. ✅ No exceptions are thrown
6. ✅ HTTP requests never reach Kokoro service
7. ✅ Dedicated clients don't fix the issue
8. ✅ Problem is 100% reproducible

## ❌ What We Know Doesn't Work

1. ❌ Shared httpx.AsyncClient (hypothesis disproven)
2. ❌ Enhanced error logging (no exceptions to catch)
3. ❌ Dedicated TTS client per job (doesn't help)

---

**Last Updated**: 2025-10-13 18:30 UTC
**Next Session**: Test sequential processing (max_parallel=1) to validate parallelism hypothesis
