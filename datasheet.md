# [cite_start]Datasheet for the Black-Box Optimization (BBO) Challenge Dataset [cite: 1]

## 1. Motivation
* [cite_start]**Task this dataset helps solve:** This dataset is created incrementally to solve a Black-Box Optimization (BBO) challenge[cite: 3]. [cite_start]The primary task is to efficiently identify the global maximum of eight distinct, unknown mathematical functions (black-box functions)[cite: 4]. [cite_start]It supports the development and evaluation of intelligent search strategies under constraints[cite: 5].
* [cite_start]**Who created it, and why?** The initial data for each function was provided as part of a university capstone project[cite: 6]. [cite_start]Subsequent data points are collaboratively collected by the student through iterative queries to the black-box functions[cite: 7]. [cite_start]The purpose is to demonstrate proficiency in machine learning optimization techniques and iterative problem-solving[cite: 8].
* [cite_start]**Funding or support:** This activity is part of an academic capstone project within a university program[cite: 9].
* [cite_start]**Other relevant details:** The project mimics real-world scenarios like hyperparameter tuning or drug discovery where function evaluations are expensive or time-consuming[cite: 10].

---

## 2. Composition
* [cite_start]**Types of data points:** Each instance consists of an input vector **x** (multi-dimensional floating-point vector) and its corresponding scalar output **y** (performance result)[cite: 12, 13].
* [cite_start]**Total number of instances:** Each function starts with 10 to 40 $(x, y)$ pairs[cite: 14]. [cite_start]The dataset grows by one pair per function per week[cite: 15]. [cite_start]After Round 8, each function contains between 18 to 48 pairs[cite: 16].
* [cite_start]**Completeness/Sample:** The dataset is a sparse, ever-growing sample of an infinite, continuous search space and is not complete[cite: 17, 18].
* [cite_start]**Formats:** All values are represented as NumPy arrays of floating-point numbers[cite: 19].
* [cite_start]**Labels/Annotations:** The **y** values serve as direct numerical labels; no additional annotation is involved[cite: 20].
* [cite_start]**Missing data:** None; every query **x** returns a **y** output[cite: 21].
* [cite_start]**Relationships between instances:** Each pair is independently observed but collectively contributes to building a Gaussian Process (GP) probabilistic surrogate model[cite: 22].
* [cite_start]**Data splits:** Not applicable for traditional train/test splits; the entire accumulated dataset is used sequentially for training at each iteration[cite: 23].
* [cite_start]**Dimensionality:** The 8 functions have increasing dimensionality from 2D to 8D[cite: 27].

---

## 3. Collection Process
* [cite_start]**Methods:** Data is gathered through sequential iterative evaluations of the eight black-box functions[cite: 30].
* [cite_start]**Goals:** To maximize the **y** output with the minimum number of queries[cite: 31].
* **Sampling strategy:**
    * [cite_start]**Initial Data:** Provided by challenge organizers[cite: 33].
    * [cite_start]**Rounds 1-3 (Heuristic Phase):** Random search for exploration and manual perturbations for exploitation[cite: 34, 35].
    * [cite_start]**Rounds 4-7 (Manual GP-BO Phase):** GP-based Bayesian Optimization using `scikit-learn`, switching between Expected Improvement (EI) and Upper Confidence Bound (UCB)[cite: 36, 37].
    * [cite_start]**Rounds 8+ (Optuna-based BO Phase):** Leveraging the `OptunaBayesianOptimizer` class for automated BO[cite: 38, 39].
* [cite_start]**Time frame:** Data is collected weekly[cite: 40].

---

## 4. Preprocessing/Cleaning/Labeling
* **Transformations applied:**
    * [cite_start]**Negation of y:** Raw **y** outputs are negated ($-y$) to convert maximization into minimization for specific BO libraries[cite: 45].
    * [cite_start]**Kernel Hyperparameter Optimization:** The GP model internally optimizes kernel hyperparameters (e.g., `length_scale`) during fitting[cite: 46].
* [cite_start]**Data cleaning:** No explicit cleaning is performed as outputs are direct from synthetic functions[cite: 47].
* [cite_start]**Raw data preserved:** Yes, the `combined_funcX_inputs` and `combined_funcX_outputs` NumPy arrays retain all raw data[cite: 48].
* [cite_start]**Intended uses:** Training Bayesian Optimization surrogate models and guiding the selection of future query points[cite: 49, 50].

---

## 5. Distribution and Maintenance
* [cite_start]**Shared with third parties:** Intended to be part of a public GitHub repository[cite: 53].
* [cite_start]**Who maintains it:** The student (myself) is responsible for updates[cite: 56].
* [cite_start]**Maintenance policies:** Version control is managed via Git and GitHub, with updates performed periodically after each round[cite: 57, 58].