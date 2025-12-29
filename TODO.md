# 🧾 Project Refactor TODO List

## How to Use This File

*   Tasks are grouped **by file**.
*   Items are **actionable and checkable**.
*   No task should be skipped.
*   The execution order outlined in `PLAN.md` should be followed.

---

## 📄 `core/execution.py`

### 🧩 File Role

Handles core logic for signal processing, trade execution, risk management, and state tracking.

---

### ❌ Identified Problems

*   Uses entry prices (`cost basis`) instead of current market prices for equity and exposure valuation.
*   Performs multi-step database writes (log trade, update position, update cash) without atomic transactions.
*   Uses a global, unseeded `np.random.normal`, making backtests non-reproducible.
*   Calculates total equity using only the price of the current signal's asset, ignoring other open positions.
*   Instance is stateful (`daily_pnl`, `consecutive_losses`), making it non-reentrant and unsafe for concurrency.
*   Silently falls back to hardcoded defaults for missing risk configuration.
*   Uses a hardcoded, inaccurate volatility value when simulating stop-loss executions.
*   (from audit) Assumes naive datetimes from DB are UTC, risking time-series errors.
*   (from audit) Daily state reset is fragile and tied to signal timestamps.
*   (from audit) Redundant calculation of position size in validation and execution steps.
*   (from audit) Market impact simulation is incomplete (hardcoded `size_multiplier`).
*   (from audit) Confusing variable naming (`amount_usd` vs. `max_total_cost`).
*   (from audit) Brittle string parsing for symbols (`.split('/')`).

---

### 🔧 TODO — Required Fixes & Changes

*   [ ] Change `_calculate_total_equity` signature to accept `market_prices: Dict[str, float]`.
*   [ ] In `_calculate_total_equity`, replace entry-price valuation with mark-to-market logic using the `market_prices` dict.
*   [ ] In `_validate_risk`, replace entry-price exposure calculation with mark-to-market logic using `market_prices`.
*   [ ] Change `get_realistic_price` signature to accept `rng: np.random.Generator`.
*   [ ] In `get_realistic_price`, replace `np.random.normal` with `rng.normal`.
*   [ ] Change `process_signal` signature to accept `market_prices: Dict[str, float]` and `rng: np.random.Generator`.
*   [ ] In `process_signal`, pass the `market_prices` dict to `_calculate_total_equity`.
*   [ ] In `process_signal`, pass the `rng` object to price simulation calls.
*   [ ] Add a `try...except...finally` block to `_execute_buy` for transaction handling.
*   [ ] Call `self.db.begin_transaction()` at the start of the `try` block in `_execute_buy`.
*   [ ] Call `self.db.commit()` at the end of the `try` block in `_execute_buy`.
*   [ ] Call `self.db.rollback()` in the `except` block of `_execute_buy`.
*   [ ] Repeat the transaction implementation for `_execute_sell`.
*   [ ] Remove instance variables: `self.consecutive_losses`, `self.daily_pnl`, `self.last_trade_date`.
*   [ ] In methods that used instance state, replace with calls to `self.db.get_strategy_state()` and `self.db.update_strategy_state()`.
*   [ ] In `check_stop_losses`, fetch the current volatility for the symbol and pass it to `_execute_sell` instead of the hardcoded `0.02`.
*   [ ] (from audit) In `_get_strategy_risk_setting`, remove the `default` parameter and raise a `ConfigurationError` if a setting is not found.
*   [ ] (from audit) Add validation to ensure all datetimes retrieved from the database are timezone-aware before use.
*   [ ] (from audit) Refactor daily state reset to be triggered by a dedicated daily event, not an incoming signal timestamp.
*   [ ] (from audit) Pass the calculated position size from `_validate_risk` to `_execute_buy` to avoid recalculation.
*   [ ] (from audit) Move hardcoded `size_multiplier` from `get_realistic_price` to `config.json`.
*   [ ] (from audit) Refactor confusing variable names (e.g., `amount_usd` to `target_notional_value`) for clarity.
*   [ ] (from audit) Replace brittle `.split('/')` logic with a more robust symbol parsing utility.

---

### 🔁 Dependency Notes

*   Changes break `main.py`, `backtest.py`, and `optimize.py`, which must be updated to call the new API.
*   Changes require `core/database.py` to be updated with transaction and state management methods.

---

### 🧪 Required Tests

*   **Test Equity Valuation**: Verify `_calculate_total_equity` uses market prices for all open positions.
*   **Test Exposure Validation**: Verify `_validate_risk` uses market prices and rejects trades correctly.
*   **Test Stop-Loss Accuracy**: Verify `check_stop_losses` passes a non-hardcoded volatility to its execution call.
*   (from audit) **Test Timezone Enforcement**: Verify the system raises an error if it encounters a naive datetime from the database.

---

## 📄 `core/database.py`

### 🧩 File Role

Provides a database interface for persisting and retrieving application data.

---

### ❌ Identified Problems

*   Lacks any support for atomic transactions, enabling state corruption.
*   Database schema is missing a table to store per-strategy state (`daily_pnl`, etc.).
*   (from audit) Singleton design without connection pooling, a potential future bottleneck.
*   (from audit) Inconsistent error handling across different methods.

---

### 🔧 TODO — Required Fixes & Changes

*   [ ] Add `begin_transaction()` method to the `Database` class.
*   [ ] Add `commit()` method to the `Database` class.
*   [ ] Add `rollback()` method to the `Database` class.
*   [ ] Add a new table `strategy_states` to `database_schema.sql`.
*   [ ] Add `get_strategy_state(strategy_id)` method to fetch data from the new table.
*   [ ] Add `update_strategy_state(strategy_id, state_data)` method to update the new table.
*   [ ] (from audit) Review and normalize exception handling to be consistent across all data access methods.
*   [ ] (from audit) Investigate replacing the singleton pattern with a connection-pooled approach for future scalability.

---

### 🔁 Dependency Notes

*   Changes are required by `core/execution.py` to fix its critical issues.

---

### 🧪 Required Tests

*   **Test Transaction Rollback**: Verify a write followed by a `rollback()` call does not persist data.
*   **Test Transaction Commit**: Verify a sequence of writes followed by a `commit()` call persists all data.
*   **Test State Management**: Verify `get_strategy_state` and `update_strategy_state` correctly read from and write to the database.

---

## 📄 `main.py` and `backtest.py`

### 🧩 File Role

Serve as the primary entry points and orchestrators for live trading and backtesting runs.

---

### ❌ Identified Problems

*   They call `ExecutionManager` using an old API that will break after refactoring.
*   They lack the logic to fetch market prices for all open positions before processing a signal.
*   (from audit) They instantiate dependencies separately, without a centralized container, leading to boilerplate.

---

### 🔧 TODO — Required Fixes & Changes

*   [ ] At startup, create a seeded `np.random.Generator` instance based on `config.json`.
*   [ ] In the main loop/backtest loop, before processing a signal:
    *   [ ] Get a list of all symbols for open positions from the database.
    *   [ ] Call `market_data` to fetch current prices for those symbols, creating a `market_prices` dictionary.
*   [ ] Update the call to `execution_manager.process_signal` to pass the `market_prices` dict and the `rng` instance.
*   [ ] (from audit) Review non-centralized dependency instantiation for potential refactoring into a shared factory function.

---

### 🔁 Dependency Notes

*   Dependent on API changes in `core/execution.py`, `core/database.py`, and `core/market_data.py`.

---

### 🧪 Required Tests

*   **Test Integration**: Verify that for one loop, market prices are fetched *before* `process_signal` is called.
*   **Test Determinism**: Verify `backtest.py` produces identical results when run twice with the same seed.

---

## 📄 `optimize.py`

### 🧩 File Role

Runs hyperparameter optimization studies.

---

### ❌ Identified Problems

*   (from audit) Not updated to handle new deterministic RNG requirements, making optimization results invalid and non-reproducible.

---

### 🔧 TODO — Required Fixes & Changes

*   [ ] (from audit) Update `optimize.py` to create and pass a correctly seeded `np.random.Generator` instance to the `ExecutionManager` for each trial.

---

### 🔁 Dependency Notes

*   Dependent on API changes in `core/execution.py`.

---

### 🧪 Required Tests

*   **Test Optimization Determinism**: Verify that a single optimization trial can be re-run with the same seed and produce the same result.

---

## 📄 `dashboard.py` and `analytics.py`

### 🧩 File Role

Provide data visualization and analysis of trading results.

---

### ❌ Identified Problems

*   (from audit) May be broken by database schema changes (e.g., new `strategy_states` table) and may rely on data that is now known to be incorrect.

---

### 🔧 TODO — Required Fixes & Changes

*   [ ] (from audit) Review all database queries in `dashboard.py` and `analytics.py` to ensure they are compatible with the updated database schema.
*   [ ] (from audit) Validate that visualizations correctly represent the new, correct data structures.

---

### 🔁 Dependency Notes

*   Dependent on schema changes in `core/database.py`.

---

## 🔗 Cross-File TODOs

*   [ ] **Propagate Market Prices**: Ensure the `market_prices: Dict[str, float]` dictionary is created in the main loop and passed through `process_signal` to all valuation functions.
*   [ ] **Enforce Deterministic RNG**: Ensure the seeded `np.random.Generator` is created in the entry points (`main.py`, `backtest.py`, `optimize.py`) and passed to `ExecutionManager` and all simulation functions.
*   [ ] **Externalize State**: Complete the removal of state variables from `ExecutionManager` and ensure state is read from and written to `core/database.py` on each transaction.
*   [ ] **Enforce Strict Configuration**: Audit all calls to `_get_strategy_risk_setting` and ensure they are wrapped in error handling for the newly possible `ConfigurationError`.

---

## 🧠 Completion Rules

*   `TODO.md` is complete only when **all checkboxes are checked**.
*   No production use before completion.
*   No optimization before completion.
*   No live trading before completion.

---
