# Routing Algo for Efficient NLP Optimisation

An ultra-fast, highly optimized Kannada OCR Post-Processing and Spell Correction module. This project demonstrates how to scale an $O(N)$ linear dictionary search bottleneck down to an $O(\log N)$ powerhouse using advanced data structures and concurrency.

## 🚀 Key Features

* **BK-Trees (Burkhard-Keller Trees):** Replaces naive linear dictionary scans with metric space pruning using Levenshtein distance, instantly eliminating irrelevant branches to achieve $O(\log N)$ search speeds.
* **LRU (Least Recently Used) Caching:** Memorizes frequent word corrections (e.g., conjunctions, common nouns) to completely bypass tree traversal on repetitive hits.
* **Parallel Processing:** Utilizes Python's `ThreadPoolExecutor` to distribute NLP correction workloads across multiple CPU cores simultaneously.
* **Adaptive Ranking:** Intelligent tie-breaking for equidistant spelling corrections based on language frequency statistics.
* **Interactive Evaluation Dashboard:** A beautiful, dependency-free Vanilla HTML/JS/CSS frontend to visualize latency comparisons, cache hits, thread utilization, and module performance breakdowns in real-time.

## 🛠️ Tech Stack

* **Backend:** Python 3, FastAPI, Uvicorn, Levenshtein
* **Frontend:** Vanilla HTML5, CSS3, JavaScript, Chart.js
* **Data:** Custom 100k+ Kannada word synthetic/real dictionary

## ⚙️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/apk-exe1/routing-algo-for-efficient-nlp-optimisation.git
   cd routing-algo-for-efficient-nlp-optimisation
   ```

2. **Install dependencies:**
   Make sure you have Python installed, then run:
   ```bash
   pip install fastapi uvicorn Levenshtein
   ```

3. **Generate the synthetic dictionary (if not present):**
   ```bash
   python generate_dictionary.py
   ```

## 🏃‍♂️ Running the Application

1. **Start the FastAPI Backend:**
   ```bash
   python -m uvicorn app:app --reload --port 8000
   ```
   The API will start running at `http://localhost:8000`.

2. **Open the Dashboard:**
   Simply double-click the `frontend/index.html` file to open it in your browser. No Node.js or web server is required for the frontend!

3. **Test the Algorithms:**
   Paste some Kannada text into the input box and click **Run Standard Algorithm** to see the baseline $O(N)$ performance, then click **Run Optimized Algorithm** to see the BK-Tree and multithreading optimizations in action.

## 📊 Presentation Document
For a deep dive into the algorithmic architecture, API payload structures, and mathematical concepts used in this project, please refer to the included `presentation.md` file.
