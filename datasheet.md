# Datasheet for the Black-Box Optimization (BBO) Challenge Dataset

## 1. Motivation
* **Task this dataset helps solve:** This dataset is created incrementally to solve a Black-Box Optimization (BBO) challenge[cite: 3]. The primary task is to efficiently identify the global maximum of eight distinct, unknown mathematical functions (black-box functions)[cite: 4]. It supports the development and evaluation of intelligent search strategies under constraints[cite: 5].
* **Who created it, and why?** The initial data for each function was provided as part of a university capstone project[cite: 6]. Subsequent data points are collaboratively collected by the student through iterative queries to the black-box functions[cite: 7]. The purpose is to demonstrate proficiency in machine learning optimization techniques and iterative problem-solving[cite: 8].
* **Funding or support:** This activity is part of an academic capstone project within a university program[cite: 9].
* **Other relevant details:** The project mimics real-world scenarios like hyperparameter tuning or drug discovery where function evaluations are expensive or time-consuming[cite: 10].

---

## 2. Composition
* **Types of data points:** Each instance consists of an input vector **x** (multi-dimensional floating-point vector) and its corresponding scalar output **y** (performance result)[cite: 12, 13].
* **Total number of instances:** Each function starts with 10 to 40 $(x, y)$ pairs[cite: 14]. The dataset grows by one pair per function per week[cite: 15]. After Round 8, each function contains between 18 to 48 pairs[cite: 16].
* **Completeness/Sample:** The dataset is a sparse, ever-growing sample of an infinite, continuous search space and is not complete[cite: 17, 18].
* **Formats:** All values are represented as NumPy arrays of floating-point numbers[cite: 19].
* **Labels/Annotations:** The **y** values serve as direct numerical labels; no additional annotation is involved[cite: 20].
* **Missing data:** None; every query **x** returns a **y** output[cite: 21].
* **Relationships between instances:** Each pair is independently observed but collectively contributes to building a Gaussian Process (GP) probabilistic surrogate model[cite: 22].
* **Data splits:** Not applicable for traditional train/test splits; the entire accumulated dataset is used sequentially for training at each iteration[cite: 23].
* **Dimensionality:** The 8 functions have increasing dimensionality from 2D to 8D[cite: 27].

---

## 3. Collection Process
* **Methods:** Data is gathered through sequential iterative evaluations of the eight black-box functions[cite: 30].
* **Goals:** To maximize the **y** output with the minimum number of queries[cite: 31].
* **Sampling strategy:**
    * **Initial Data:** Provided by challenge organizers[cite: 33].
    * **Rounds 1-3 (Heuristic Phase):** Random search for exploration and manual perturbations for exploitation[cite: 34, 35].
    * **Rounds 4-7 (Manual GP-BO Phase):** GP-based Bayesian Optimization using `scikit-learn`, switching between Expected Improvement (EI) and Upper Confidence Bound (UCB)[cite: 36, 37].
    * **Rounds 8+ (Optuna-based BO Phase):** Leveraging the `OptunaBayesianOptimizer` class for automated BO[cite: 38, 39].
* **Time frame:** Data is collected weekly[cite: 40].

---

## 4. Preprocessing/Cleaning/Labeling
* **Transformations applied:**
    * **Negation of y:** Raw **y** outputs are negated ($-y$) to convert maximization into minimization for specific BO libraries[cite: 45].
    * **Kernel Hyperparameter Optimization:** The GP model internally optimizes kernel hyperparameters (e.g., `length_scale`) during fitting[cite: 46].
* **Data cleaning:** No explicit cleaning is performed as outputs are direct from synthetic functions[cite: 47].
* **Raw data preserved:** Yes, the `combined_funcX_inputs` and `combined_funcX_outputs` NumPy arrays retain all raw data[cite: 48].
* **Intended uses:** Training Bayesian Optimization surrogate models and guiding the selection of future query points[cite: 49, 50].

---

## 5. Distribution and Maintenance
* **Shared with third parties:** Intended to be part of a public GitHub repository[cite: 53].
* **Who maintains it:** The student (myself) is responsible for updates[cite: 56].
* **Maintenance policies:** Version control is managed via Git and GitHub, with updates performed periodically after each round[cite: 57, 58].