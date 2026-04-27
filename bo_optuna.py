import optuna
from optuna.samplers import TPESampler
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import warnings
warnings.filterwarnings('ignore')


class OptunaBayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Process surrogate with Optuna for hyperparameter tuning
    and acquisition function optimization.
    
    Uses:
    - Gaussian Process (GP) as surrogate model
    - Selectable acquisition functions: UCB, EI, Log EI, PI
    - Optuna TPE sampler to optimize the acquisition function
    - (Optional) Optuna to tune GP hyperparameters and the active acquisition parameter
    """

    SUPPORTED_ACQUISITIONS = {
        'ucb': 'UCB',
        'ei': 'EI',
        'log_ei': 'Log EI',
        'pi': 'PI',
    }
    
    def __init__(self, X_initial, y_initial, bounds, optimize_hp=True, random_state=42,
                 acquisition='ucb', hp={}):
        """
        Initialize the Bayesian Optimizer.
        
        Parameters:
        -----------
        X_initial : array-like, shape (n_samples, n_features)
            Initial training data (inputs)
        y_initial : array-like, shape (n_samples,)
            Initial training data (outputs)
        bounds : list of tuples
            Search space bounds [(x1_min, x1_max), (x2_min, x2_max), ...]
        optimize_hp : bool, default=True
            Whether to enable hyperparameter optimization
        random_state : int, default=42
            Random seed for reproducibility
        acquisition : str, default='ucb'
            Acquisition function to optimize: 'ucb', 'ei', 'log_ei', or 'pi'
        """
        self.X_train = np.asarray(X_initial).reshape(-1, len(bounds))
        self.y_train = np.asarray(y_initial).reshape(-1)
        self.bounds = bounds
        self.optimize_hp = optimize_hp
        self.random_state = random_state
        self.n_features = len(bounds)
        self.acquisition = self._normalize_acquisition(acquisition)
        
        # Initialize hyperparameters
        self.hp = {
            'constant': 1.0,
            'length_scale': 0.5,
            'kappa': 15.0,
            'xi': 0.01,
            'alpha': 1e-6
        }
        
        # History tracking
        self.history = {
            'X_candidates': [],
            'acquisitions': [],
            'hyperparameters': [],
            'best_acq_value': -np.inf
        }
        
        # Optuna study storage (for visualization)
        self.acquisition_studies = []  # List of acquisition optimization studies
        self.hp_studies = []           # List of hyperparameter optimization studies
        
        # Current GP (will be fitted on demand)
        self.gp = None
        
        np.random.seed(random_state)

    def _normalize_acquisition(self, acquisition):
        """
        Validate and normalize the acquisition name.
        """
        normalized = acquisition.strip().lower().replace('-', '_').replace(' ', '_')
        aliases = {
            'upper_confidence_bound': 'ucb',
            'expected_improvement': 'ei',
            'logei': 'log_ei',
            'log_expected_improvement': 'log_ei',
            'probability_of_improvement': 'pi',
        }
        normalized = aliases.get(normalized, normalized)

        if normalized not in self.SUPPORTED_ACQUISITIONS:
            valid = ', '.join(sorted(self.SUPPORTED_ACQUISITIONS))
            raise ValueError(f"Unsupported acquisition '{acquisition}'. Choose from: {valid}.")

        return normalized

    def _acquisition_label(self):
        """
        Human-readable label for the active acquisition function.
        """
        return self.SUPPORTED_ACQUISITIONS[self.acquisition]

    def _active_acquisition_param_name(self):
        """
        Acquisition-specific exploration parameter name.
        """
        return 'kappa' if self.acquisition == 'ucb' else 'xi'

    def _resolve_acquisition_params(self, acquisition_params=None):
        """
        Merge explicit acquisition parameters with the optimizer defaults.
        """
        params = {
            'kappa': self.hp['kappa'],
            'xi': self.hp['xi'],
        }
        if acquisition_params is not None:
            params.update(acquisition_params)
        return params

    def _expected_improvement(self, mu, sigma, y_best, xi):
        """
        Expected Improvement for a maximization objective.
        """
        improvement = mu - y_best - xi
        if sigma <= 1e-12:
            return max(improvement, 0.0)

        z_score = improvement / sigma
        return improvement * norm.cdf(z_score) + sigma * norm.pdf(z_score)

    def _probability_of_improvement(self, mu, sigma, y_best, xi):
        """
        Probability of Improvement for a maximization objective.
        """
        improvement = mu - y_best - xi
        if sigma <= 1e-12:
            return 1.0 if improvement > 0.0 else 0.0

        z_score = improvement / sigma
        return norm.cdf(z_score)
    
    def _fit_gp(self, constant=None, length_scale=None, alpha=None):
        """
        Fit Gaussian Process with specified hyperparameters.
        
        Parameters:
        -----------
        constant : float, optional
            ConstantKernel scaling parameter
        length_scale : float, optional
            RBF kernel length scale
        alpha : float, optional
            GP noise level
            
        Returns:
        --------
        gp : fitted GaussianProcessRegressor
        """
        if constant is None:
            constant = self.hp['constant']
        if length_scale is None:
            length_scale = self.hp['length_scale']
        if alpha is None:
            alpha = self.hp['alpha']
        
        # Build kernel
        const_kernel = ConstantKernel(constant, constant_value_bounds=(1e-3, 1e3))
        rbf_kernel = RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))
        kernel = const_kernel * rbf_kernel
        
        # Fit GP
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=10,
            normalize_y=False
        )
        gp.fit(self.X_train, self.y_train)
        
        return gp
    
    def _acquisition_function(self, X, gp, acquisition_params=None):
        """
        Compute the active acquisition function.
        
        Supported functions for maximization:
        - UCB(x) = mu(x) + kappa * sigma(x)
        - EI(x) = E[max(f(x) - f_best - xi, 0)]
        - Log EI(x) = log(EI(x))
        - PI(x) = P(f(x) >= f_best + xi)
        
        Parameters:
        -----------
        X : array-like, shape (n_features,)
            Point to evaluate
        gp : GaussianProcessRegressor
            Fitted GP model
        acquisition_params : dict, optional
            Overrides for acquisition-specific parameters
            
        Returns:
        --------
        float : acquisition value
        """
        params = self._resolve_acquisition_params(acquisition_params)
        X = np.asarray(X).reshape(1, -1)
        mu, sigma = gp.predict(X, return_std=True)
        mu = float(mu[0])
        sigma = float(max(sigma[0], 0.0))

        if self.acquisition == 'ucb':
            return mu + params['kappa'] * sigma

        y_best = float(np.max(self.y_train))
        if self.acquisition == 'ei':
            return self._expected_improvement(mu, sigma, y_best, params['xi'])
        if self.acquisition == 'log_ei':
            ei = self._expected_improvement(mu, sigma, y_best, params['xi'])
            return float(np.log(np.clip(ei, 1e-300, None)))
        if self.acquisition == 'pi':
            return self._probability_of_improvement(mu, sigma, y_best, params['xi'])

        raise RuntimeError(f"Unhandled acquisition function '{self.acquisition}'.")
    
    def _optimize_acquisition_with_optuna(self, gp, n_trials=100, acquisition_params=None,
                                          store_study=True):
        """
        Use Optuna TPE sampler to find the best point maximizing acquisition function.
        
        Parameters:
        -----------
        gp : GaussianProcessRegressor
            Fitted GP model
        n_trials : int, default=100
            Number of Optuna trials for acquisition optimization
        acquisition_params : dict, optional
            Overrides for acquisition-specific parameters
        store_study : bool, default=True
            Whether to store the Optuna study for later visualization
            
        Returns:
        --------
        X_next : array, shape (n_features,)
            Proposed next evaluation point
        best_acq : float
            Best acquisition value found
        """
        def optuna_objective(trial):
            # Suggest x1, x2, ... parameters within bounds
            X_suggest = np.array([
                trial.suggest_float(f"x{i}", self.bounds[i][0], self.bounds[i][1])
                for i in range(self.n_features)
            ])
            
            # Evaluate acquisition function
            acq_val = self._acquisition_function(
                X_suggest, gp, acquisition_params=acquisition_params
            )
            return acq_val
        
        # Create study with TPE sampler
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(optuna_objective, n_trials=n_trials, show_progress_bar=False)
        
        # Store study for visualization
        if store_study:
            self.acquisition_studies.append(study)
        
        # Extract best parameters and value
        best_trial = study.best_trial
        X_next = np.array([best_trial.params[f"x{i}"] for i in range(self.n_features)])
        best_acq = best_trial.value
        
        return X_next, best_acq
    
    def _optimize_hyperparameters(self, n_trials=50):
        """
        Use Optuna to tune GP hyperparameters and the active acquisition parameter.
        Objective: maximize the acquisition function value.
        
        Parameters:
        -----------
        n_trials : int, default=50
            Number of Optuna trials for hyperparameter tuning
            
        Returns:
        --------
        best_hp : dict
            Best hyperparameters found
        """
        def hp_objective(trial):
            # Suggest hyperparameter values
            constant = trial.suggest_float('constant', 0.1, 10.0, log=True)
            length_scale = trial.suggest_float('length_scale', 1e-3, 10.0, log=True)
            alpha = trial.suggest_float('alpha', 1e-8, 1e-3, log=True)
            acquisition_params = None

            if self.acquisition == 'ucb':
                acquisition_params = {
                    'kappa': trial.suggest_float('kappa', 0.1, 20.0)
                }
            else:
                acquisition_params = {
                    'xi': trial.suggest_float('xi', 1e-4, 1.0, log=True)
                }
            
            try:
                # Fit GP with these hyperparameters
                gp_temp = self._fit_gp(constant=constant, length_scale=length_scale, alpha=alpha)
                
                # Optimize acquisition with these hyperparameters
                _, acq_value = self._optimize_acquisition_with_optuna(
                    gp_temp,
                    n_trials=20,
                    acquisition_params=acquisition_params,
                    store_study=False
                )
                
                # Use the best acquisition value directly as the objective.
                return acq_value
            except Exception as e:
                # If GP fitting fails, return worst possible value
                return -np.inf
        
        # Create study
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(hp_objective, n_trials=n_trials, show_progress_bar=False)
        
        # Store study for visualization
        self.hp_studies.append(study)
        
        # Extract best hyperparameters
        best_trial = study.best_trial
        best_hp = {
            'constant': best_trial.params['constant'],
            'length_scale': best_trial.params['length_scale'],
            'kappa': self.hp['kappa'],
            'xi': self.hp['xi'],
            'alpha': best_trial.params['alpha']
        }

        if 'kappa' in best_trial.params:
            best_hp['kappa'] = best_trial.params['kappa']
        if 'xi' in best_trial.params:
            best_hp['xi'] = best_trial.params['xi']
        
        return best_hp
    
    def optimize(self, n_iterations=5, optimize_hp_every=3, optimize_hp_n_trials=30, 
                 acq_n_trials=100, verbose=True):
        """
        Main optimization loop proposing next evaluation candidates.
        
        Parameters:
        -----------
        n_iterations : int, default=5
            Number of BO iterations
        optimize_hp_every : int, default=3
            Tune hyperparameters every N iterations (0 to disable)
        optimize_hp_n_trials : int, default=30
            Optuna trials for hyperparameter optimization
        acq_n_trials : int, default=100
            Optuna trials for acquisition function optimization
        verbose : bool, default=True
            Print progress
            
        Returns:
        --------
        proposals : list of arrays
            Proposed evaluation points
        """
        proposals = []
        
        for iteration in range(n_iterations):
            # Optionally tune hyperparameters
            if optimize_hp_every > 0 and iteration % optimize_hp_every == 0 and iteration > 0 and self.optimize_hp:
                if verbose:
                    print(f"[Iteration {iteration}] Optimizing hyperparameters...")
                best_hp = self._optimize_hyperparameters(n_trials=optimize_hp_n_trials)
                self.hp = best_hp
                active_param = self._active_acquisition_param_name()
                if verbose:
                    print(f"  Best hp: constant={best_hp['constant']:.4f}, length_scale={best_hp['length_scale']:.4f}, "
                          f"{active_param}={best_hp[active_param]:.4f}, alpha={best_hp['alpha']:.2e}")
            
            # Fit GP with current hyperparameters
            self.gp = self._fit_gp(
                constant=self.hp['constant'],
                length_scale=self.hp['length_scale'],
                alpha=self.hp['alpha']
            )
            
            # Optimize acquisition function
            X_next, best_acq = self._optimize_acquisition_with_optuna(
                self.gp, n_trials=acq_n_trials
            )
            
            # Track proposal
            proposals.append(X_next)
            self.history['X_candidates'].append(X_next)
            self.history['acquisitions'].append(best_acq)
            self.history['hyperparameters'].append(self.hp.copy())
            self.history['best_acq_value'] = max(self.history['best_acq_value'], best_acq)
            
            if verbose:
                print(
                    f"[Iteration {iteration}] Proposed: {X_next}, "
                    f"{self._acquisition_label()}: {best_acq:.6f}"
                )
        
        return proposals
    
    def update(self, x_new, y_new):
        """
        Update training data with new observation.
        
        Parameters:
        -----------
        x_new : array-like
            New input point
        y_new : float
            New output value
        """
        x_new = np.asarray(x_new).reshape(1, -1)
        self.X_train = np.vstack([self.X_train, x_new])
        self.y_train = np.concatenate([self.y_train, [y_new]])
    
    def get_best_observation(self):
        """
        Returns the best observation found so far.
        
        Returns:
        --------
        X_best, y_best
        """
        best_idx = np.argmax(self.y_train)
        return self.X_train[best_idx], self.y_train[best_idx]
    
    def get_history(self):
        """
        Returns optimization history.
        
        Returns:
        --------
        dict : history with X_candidates, acquisitions, hyperparameters
        """
        return self.history
    
    def plot_acquisition_optimization(self, iteration=None):
        """
        Visualize Optuna study for acquisition function optimization.
        
        Parameters:
        -----------
        iteration : int, optional
            Which iteration's acquisition study to plot. If None, plots the most recent.
            
        Returns:
        --------
        matplotlib figure or None if no studies available
        """
        if not self.acquisition_studies:
            print("No acquisition optimization studies to visualize.")
            return None
        
        if iteration is None:
            iteration = len(self.acquisition_studies) - 1
        
        study = self.acquisition_studies[iteration]
        
        try:
            from optuna.visualization import plot_optimization_history
            result = plot_optimization_history(study)
            # Handle both Optuna Figure objects and direct Plotly figures
            if hasattr(result, 'to_plotly_figure'):
                fig = result.to_plotly_figure()
            else:
                fig = result
            fig.update_layout(title=f"Acquisition Optimization - Iteration {iteration}")
            return fig
        except Exception as e:
            print(f"Error plotting acquisition optimization: {e}")
            return None
    
    def plot_hyperparameter_optimization(self, iteration=None):
        """
        Visualize Optuna study for hyperparameter optimization.
        
        Parameters:
        -----------
        iteration : int, optional
            Which iteration's hyperparameter study to plot. If None, plots the most recent.
            
        Returns:
        --------
        matplotlib figure or None if no studies available
        """
        if not self.hp_studies:
            print("No hyperparameter optimization studies to visualize.")
            return None
        
        if iteration is None:
            iteration = len(self.hp_studies) - 1
        
        study = self.hp_studies[iteration]
        
        try:
            from optuna.visualization import plot_optimization_history
            result = plot_optimization_history(study)
            # Handle both Optuna Figure objects and direct Plotly figures
            if hasattr(result, 'to_plotly_figure'):
                fig = result.to_plotly_figure()
            else:
                fig = result
            fig.update_layout(title=f"Hyperparameter Optimization - Iteration {iteration}")
            return fig
        except Exception as e:
            print(f"Error plotting hyperparameter optimization: {e}")
            return None
    
    def plot_param_importances(self, study_type='acquisition', iteration=None):
        """
        Visualize parameter importances using Optuna's built-in analysis.
        
        Parameters:
        -----------
        study_type : str, default='acquisition'
            'acquisition' or 'hyperparameter'
        iteration : int, optional
            Which iteration's study to analyze. If None, uses most recent.
            
        Returns:
        --------
        matplotlib figure or None if no studies available
        """
        if study_type == 'acquisition':
            if not self.acquisition_studies:
                print("No acquisition optimization studies available.")
                return None
            if iteration is None:
                iteration = len(self.acquisition_studies) - 1
            study = self.acquisition_studies[iteration]
        else:
            if not self.hp_studies:
                print("No hyperparameter optimization studies available.")
                return None
            if iteration is None:
                iteration = len(self.hp_studies) - 1
            study = self.hp_studies[iteration]
        
        try:
            from optuna.visualization import plot_param_importances
            result = plot_param_importances(study)
            # Handle both Optuna Figure objects and direct Plotly figures
            if hasattr(result, 'to_plotly_figure'):
                fig = result.to_plotly_figure()
            else:
                fig = result
            fig.update_layout(title=f"Parameter Importances ({study_type}) - Iteration {iteration}")
            return fig
        except Exception as e:
            print(f"Error plotting param importances: {e}")
            return None
    
    def plot_parallel_coordinate(self, study_type='acquisition', iteration=None):
        """
        Visualize parallel coordinates plot for trial parameters and values.
        
        Parameters:
        -----------
        study_type : str, default='acquisition'
            'acquisition' or 'hyperparameter'
        iteration : int, optional
            Which iteration's study to visualize. If None, uses most recent.
            
        Returns:
        --------
        matplotlib figure or None if no studies available
        """
        if study_type == 'acquisition':
            if not self.acquisition_studies:
                print("No acquisition optimization studies available.")
                return None
            if iteration is None:
                iteration = len(self.acquisition_studies) - 1
            study = self.acquisition_studies[iteration]
        else:
            if not self.hp_studies:
                print("No hyperparameter optimization studies available.")
                return None
            if iteration is None:
                iteration = len(self.hp_studies) - 1
            study = self.hp_studies[iteration]
        
        try:
            from optuna.visualization import plot_parallel_coordinate
            result = plot_parallel_coordinate(study)
            # Handle both Optuna Figure objects and direct Plotly figures
            if hasattr(result, 'to_plotly_figure'):
                fig = result.to_plotly_figure()
            else:
                fig = result
            fig.update_layout(title=f"Parallel Coordinates ({study_type}) - Iteration {iteration}")
            return fig
        except Exception as e:
            print(f"Error plotting parallel coordinates: {e}")
            return None
    
    def plot_slice(self, study_type='acquisition', iteration=None):
        """
        Visualize slice plot showing trial parameters vs objective value.
        
        Parameters:
        -----------
        study_type : str, default='acquisition'
            'acquisition' or 'hyperparameter'
        iteration : int, optional
            Which iteration's study to visualize. If None, uses most recent.
            
        Returns:
        --------
        matplotlib figure or None if no studies available
        """
        if study_type == 'acquisition':
            if not self.acquisition_studies:
                print("No acquisition optimization studies available.")
                return None
            if iteration is None:
                iteration = len(self.acquisition_studies) - 1
            study = self.acquisition_studies[iteration]
        else:
            if not self.hp_studies:
                print("No hyperparameter optimization studies available.")
                return None
            if iteration is None:
                iteration = len(self.hp_studies) - 1
            study = self.hp_studies[iteration]
        
        try:
            from optuna.visualization import plot_slice
            result = plot_slice(study)
            # Handle both Optuna Figure objects and direct Plotly figures
            if hasattr(result, 'to_plotly_figure'):
                fig = result.to_plotly_figure()
            else:
                fig = result
            fig.update_layout(title=f"Slice Plot ({study_type}) - Iteration {iteration}")
            return fig
        except Exception as e:
            print(f"Error plotting slice: {e}")
            return None
    
    def get_acquisition_study(self, iteration=None):
        """
        Get the raw Optuna study object for acquisition optimization.
        
        Parameters:
        -----------
        iteration : int, optional
            Which iteration's study to retrieve. If None, returns most recent.
            
        Returns:
        --------
        optuna.study.Study or None
        """
        if not self.acquisition_studies:
            return None
        if iteration is None:
            iteration = len(self.acquisition_studies) - 1
        return self.acquisition_studies[iteration]
    
    def get_hp_study(self, iteration=None):
        """
        Get the raw Optuna study object for hyperparameter optimization.
        
        Parameters:
        -----------
        iteration : int, optional
            Which iteration's study to retrieve. If None, returns most recent.
            
        Returns:
        --------
        optuna.study.Study or None
        """
        if not self.hp_studies:
            return None
        if iteration is None:
            iteration = len(self.hp_studies) - 1
        return self.hp_studies[iteration]
