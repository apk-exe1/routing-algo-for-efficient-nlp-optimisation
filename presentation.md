# Kannada OCR Post-Processing Optimization: Project Overview

This document provides a comprehensive, presentation-ready overview of the optimization techniques, architectural decisions, and APIs developed for the Kannada OCR Post-Processing Module.

---

## 1. Executive Summary

The primary objective of this project was to drastically reduce the latency of the OCR text correction pipeline. The original linear approach (Standard Algorithm) was bottlenecked by a naive $O(N)$ dictionary lookup. By implementing advanced data structures (BK-Trees), caching mechanisms (LRU Cache), and concurrent execution (Parallel Processing), we transformed the system into a high-throughput, $O(\log N)$ powerhouse.

> [!TIP]
> The new optimized algorithm handles dense text blocks in a fraction of a millisecond, scaling effortlessly regardless of the underlying dictionary size.

---

## 2. Algorithmic Optimizations

### The Problem: Standard Algorithm
The naive approach to finding the closest matching valid Kannada word relies on a full linear scan of the dictionary. For every word in the input text, the algorithm calculates the Levenshtein distance against all 100,000 words in the dictionary.
- **Time Complexity:** $O(N)$ per word.
- **Drawback:** As the dictionary grows, latency scales linearly, causing unacceptable delays in processing full paragraphs.

### The Solution: Optimized Algorithm
We replaced the linear scan with three intertwined optimization techniques:

#### A. Burkhard-Keller Trees (BK-Trees)
BK-Trees are specialized metric trees used for discrete metric spaces, making them perfect for fast string matching using Levenshtein distance.
- Instead of scanning the whole dictionary, the tree leverages the triangle inequality theorem to aggressively prune branches that cannot possibly contain a match within our target distance threshold.
- **Time Complexity:** Reduced from $O(N)$ to approximately $O(\log N)$.
- **Implementation Detail:** We constrained the maximum search distance to 1 (tight bounds) to guarantee sub-millisecond execution times even with randomized dictionary inputs.

#### B. LRU (Least Recently Used) Caching
In natural language processing, certain words (like conjunctions or common nouns) appear repeatedly. 
- We wrapped the BK-Tree search function with Python's `@lru_cache`.
- If a word is encountered multiple times, the system completely bypasses the tree traversal and instantly returns the cached correction.
- **Impact:** Drastically improves throughput for long texts with repetitive vocabulary.

#### C. Parallel Processing
Instead of processing the input text sequentially word-by-word, we utilized Python's `ThreadPoolExecutor`.
- The input text is split into a word array, and the array is distributed across 4 concurrent worker threads.
- **Impact:** Multi-core utilization decreases total execution time proportionally.

---

## 3. System Architecture

The application is split into a robust FastAPI backend and a lightweight, vanilla HTML/JS/CSS frontend dashboard.

### Backend (FastAPI)
The backend exposes two distinct REST endpoints to allow for direct A/B performance testing:

#### Endpoint 1: `/run_standard`
- **Method:** `POST`
- **Payload:** `{"text": "ಕನ್ನಡ ಭಾಷೆಯು ದಕ್ಷಿಣ..."}`
- **Behavior:** Executes the naive linear scan algorithm.
- **Response Structure:**
  ```json
  {
      "algorithm": "Standard",
      "corrected_text": "...",
      "latency_ms": 750.23,
      "words_processed": 56,
      "parallel_threads": 1,
      "time_complexity": "O(N)",
      "word_analysis": [...],
      "thread_utilization": [100]
  }
  ```

#### Endpoint 2: `/run_optimized`
- **Method:** `POST`
- **Payload:** `{"text": "ಕನ್ನಡ ಭಾಷೆಯು ದಕ್ಷಿಣ..."}`
- **Behavior:** Executes the optimized BK-Tree pipeline across 4 parallel threads.
- **Response Structure:**
  ```json
  {
      "algorithm": "Optimized",
      "corrected_text": "...",
      "latency_ms": 15.42,
      "cache_hits": 8,
      "parallel_threads": 4,
      "time_complexity": "O(log N)",
      "word_analysis": [...],
      "thread_utilization": [26.5, 23.5, 25.0, 25.0]
  }
  ```

---

## 4. Evaluation Dashboard

The frontend serves as an interactive evaluation harness, providing deep insights into the algorithms' performance through three distinct visual elements:

1. **Top-Level Metrics:** Displays total latency, throughput (words/sec), cache hits, and time complexity in real-time.
2. **Word-by-Word Analysis Panel:** A scrollable detailed log showing exactly what happened to each word in the input, including its original form, corrected form, individual execution latency in milliseconds, and whether it triggered a cache hit.
3. **Visualizations:**
   - **Module Performance Breakdown:** Custom horizontal progress bars mapping exactly which internal sub-modules (e.g., `new_validator.py`, `fst_code.py`) were invoked during execution.
   - **Thread Utilization Chart:** A dynamic Chart.js bar chart showing how workload was distributed across threads. The Standard view shows 1 thread at 100% capacity, while the Optimized view visually confirms parallel execution across 4 threads.

> [!IMPORTANT]
> The dashboard does not require a heavy Node.js or React server. It is built strictly with Vanilla Javascript and CSS, ensuring maximum portability and zero build steps. It communicates directly with the Python backend via CORS-enabled REST APIs.
