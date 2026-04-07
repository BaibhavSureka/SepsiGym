# COMPREHENSIVE CODE VERIFICATION & ENHANCEMENT PROMPT

You are a senior software engineer reviewing production code for a medical AI system evaluation platform.

## TASK

Review the provided `inference.py` implementation and:

1. Identify ALL potential failure modes
2. Verify it passes Phase 1 (correctness) AND Phase 2 (robustness)
3. Enhance code to handle edge cases
4. Ensure NO unhandled exceptions can occur
5. Verify output JSON structure is always valid

---

## PHASE 1 CRITERIA (Correctness)

✓ Code runs without syntax errors
✓ Imports all required packages
✓ Policy generates valid SepsisAction objects
✓ Environment interactions work (reset, step, close)
✓ JSON output is valid and contains all required fields
✓ Metrics are correctly extracted from env responses
✓ Episode loops complete without crashes

### Phase 1 Tests:

```bash
python -m py_compile inference.py  # No syntax errors
python inference.py --episodes=1   # Single episode completes
python inference.py --episodes=3 --model=auto  # Auto mode works
```

---

## PHASE 2 CRITERIA (Robustness - FAIL-FAST)

❌ Phase 2 fails on ANY unhandled exception
❌ Must never exit with non-zero status
❌ Must handle ALL error conditions gracefully

### Critical Failure Points to Fix:

**1. Environment Initialization**

- [ ] Env connection fails (host unreachable)
- [ ] Env timeout (slow response)
- [ ] Invalid base_url or task_id
- **FIX**: Wrap in try/except, return sensible default

**2. Step Execution Loop**

- [ ] env.step() returns None
- [ ] action object creation fails
- [ ] observation parsing fails
- [ ] Reward is NaN or invalid type
- **FIX**: Validate each return value, catch exceptions

**3. State Query & Cleanup**

- [ ] env.state() throws exception
- [ ] env.close() throws exception
- [ ] state object missing required attributes
- **FIX**: Defensive access, fallback objects

**4. Metrics Extraction**

- [ ] final_info is None or empty dict
- [ ] metrics missing expected keys
- [ ] Score is NaN, None, or unparseable
- **FIX**: Use .get() with defaults, type conversion in try/except

**5. Result Dictionary Construction**

- [ ] Missing required keys in return dict
- [ ] compute_dense_reward_metrics fails
- [ ] Policy source aggregation fails
- **FIX**: Return complete dict even on error, all keys guaranteed

**6. Main Loop**

- [ ] Episode list comprehension fails on first task
- [ ] summarize_runs() receives incomplete results
- [ ] JSON serialization fails
- [ ] Output file write fails
- **FIX**: Episode-level try/except, defensive .get() access

**7. API Calls**

- [ ] OpenAI client initialization fails
- [ ] LLM policy generation fails
- [ ] Network timeout during inference
- **FIX**: Graceful fallback to heuristic

---

## REQUIRED FIXES

### 1. Defensive State Object

```python
# When env.state() fails or env is None:
state = type('obj', (object,), {
    'episode_id': 'unknown',
    'step_count': step_count,
    'outcome': 'failed'
})()
```

### 2. Guaranteed Return Dict Fields

Every `run_task()` must return dict with these keys (even on error):

- task_id, episode_id, score
- steps_taken, reward_count, positive_rewards_count
- safety_violations, reward_density
- policy_error_count, policy_last_error
- policy_sources, policy_mode
- avg_reward, detection, lab_workup, treatment
- timeliness, stability, safety, outcome
- steps, total_reward, avg_reward_per_step
- reward_variance, max_single_reward
- episode_length_efficiency, positive_reward_ratio
- unique_actions, action_entropy

### 3. Safe Aggregation in main()

```python
# Defensive access to all result fields:
sum(item.get("steps_taken", 0) for item in episode_results)
np.mean([item.get("score", 0.0) for item in episode_results])
```

### 4. Exception Handlers at Each Level

- ✓ Environment init: try/except
- ✓ Step loop: try/except with continue
- ✓ Value function updates: try/except
- ✓ Metrics extraction: try/except
- ✓ Result construction: try/except
- ✓ Episode loop: try/except with continue
- ✓ Main function: top-level try/except/finally

### 5. Stderr Logging

```python
import sys
print("[ERROR] description", file=sys.stderr)
# Not stdout — validator expects clean stdout
```

---

## VERIFICATION CHECKLIST

### Code Structure

- [ ] All imports present and valid
- [ ] No undefined variables
- [ ] All functions return expected types
- [ ] No infinite loops or missed breaks

### Exception Handling

- [ ] No operations outside try/except that can fail:
  - Network calls
  - Dict/list access
  - Type conversions
  - File I/O
- [ ] All exceptions caught and logged
- [ ] Graceful fallbacks for each error

### Data Flow

- [ ] Episode results always have all required keys
- [ ] summarize_runs() can handle missing fields
- [ ] JSON serialization never fails
- [ ] Output file path is always writable

### Edge Cases

- [ ] Empty episodes list → handled
- [ ] Zero steps taken → handled
- [ ] NaN metrics → handled
- [ ] Missing observations → handled
- [ ] Concurrent errors → handled

---

## TESTING SCENARIOS

Before submission, test these locally:

```bash
# Test 1: Basic run
python inference.py --episodes=1

# Test 2: Multiple episodes
python inference.py --episodes=3

# Test 3: Auto policy selection
python inference.py --episodes=1 --model=auto

# Test 4: Custom output path
python inference.py --episodes=1 --output test_output.json

# Test 5: Syntax validation
python -m py_compile inference.py
```

**Expected result**: All tests complete WITHOUT exit code error, JSON output valid

---

## FINAL CHECKLIST - BEFORE SUBMISSION

**Phase 1 (Correctness)**

- [ ] `python -m py_compile inference.py` returns 0
- [ ] `python inference.py --episodes=1` completes
- [ ] Output JSON is valid and parseable
- [ ] No imports fail on first line
- [ ] All functions defined before use

**Phase 2 (Robustness)**

- [ ] Exit code is 0 (even on env connection fail)
- [ ] No unhandled exceptions in stderr
- [ ] Every run_task() returns complete result dict
- [ ] main() never raises exception to validator
- [ ] Graceful handling of:
  - Environment unreachable
  - Slow/timeout responses
  - Invalid observations
  - Missing metrics
  - Corrupted state

**Submission Readiness**

- [ ] Git commits pushed to main
- [ ] HuggingFace space synced
- [ ] All test runs successful locally
- [ ] No debug print statements
- [ ] Proper error logging to stderr

---

## PROMPT TO CLAUDE/CODEX

"Review this SepsiGym inference.py code and make these changes:

1. **Wrap ALL risky operations in try/except**:
   - Environment initialization
   - env.step() calls
   - Value function updates
   - Metrics extraction
   - Result dict construction

2. **Guarantee complete result dictionary** with fallback values for ALL 25+ expected keys even if everything fails

3. **Add defensive .get() access** in summarize_runs() to handle missing result fields

4. **Wrap main() episode loop** in try/except to prevent one failed task from crashing all episodes

5. **Add top-level exception handler** in main() with stderr logging

6. **Ensure env.close() always runs** via finally block, even if env.state() fails

7. **Return sensible defaults** for:
   - state object when env.state() fails
   - metrics dict when extraction fails
   - Everything when env initialization fails

8. **Test these scenarios**:

   ```
   - Environment connection fails
   - env.step() times out
   - Metrics missing from response
   - Observer state corrupted
   - Zero steps completed
   ```

9. **Verify**:
   - No syntax errors
   - Exit code is 0 for all runs
   - JSON output always valid
   - All required keys in output

IMPORTANT: This code is evaluated by a strict validator. Phase 2 is fail-fast — ANY unhandled exception fails the entire evaluation. Make it bulletproof."

---

## INTEGRATION WITH YOUR CURRENT CODE

The new advanced features are GOOD:

- ✅ Monte Carlo planning
- ✅ Beam search
- ✅ Value function learning
- ✅ Safety override
- ✅ Candidate generation

But they need exception protection:

```python
try:
    best_action = choose_action(...)
except Exception as e:
    policy_errors.append(str(e))
    best_action = heuristic_action(obs)  # Fallback
```

---

## SUBMISSION WORKFLOW

After Claude modifies code:

1. **Local test** via terminal:

   ```bash
   python -m py_compile inference.py
   python inference.py --episodes=1
   ```

2. **Git push**:

   ```bash
   git add inference.py
   git commit -m "Final: Bulletproof exception handling for Phase 1+2"
   git push origin main
   ```

3. **Submit** via platform

4. **Monitor logs** for any Phase 2 failures

---

## SUCCESS CRITERIA

✅ Phase 1: PASSED (correct output)
✅ Phase 2: PASSED (no crashes)
✅ Metrics: Reasonable scores (>0.5 per episode)
✅ Ready for Phase 3: Advanced reasoning
