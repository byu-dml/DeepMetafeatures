import numpy as np
import time
import os
import matplotlib.pyplot as plt

from dna.metrics import (
    n_correct_at_k, ndcg_at_k, pearson_correlation, regret_at_k, rmse, spearman_correlation, pearson_loss_metric
)
from dna import utils


class ProblemBase:

    def __init__(self, group_key):
        self.group_key = group_key
        self.group_scores_key = 'scores_by_{}'.format(self.group_key)
        self.agg_scores_key = 'aggregate_scores'
        self._fit_method_name = 'fit'
        self.total_scores_key = 'total_scores'
        self._predict_method_name = None

    def _validate_model_has_method(self, model, method_name):
        if not hasattr(model, method_name):
            raise ValueError(
                '{} is not designed for the {} problem. It is missing a {} method'.format(
                    model, type(self).__name__, method_name
                )
            )

    def fit(
        self, train_data, model, model_config, *, refit_model=False, verbose=False, model_output_dir=None
    ):
        self._validate_model_has_method(model, self._fit_method_name)

        model_fit_config = model_config.get(self._fit_method_name, {})
        model_fit_method = getattr(model, self._fit_method_name)

        fit_time = None
        if not model.fitted or refit_model:
            start_time = time.time()
            model_fit_method(
                train_data, verbose=verbose, output_dir=model_output_dir, **model_fit_config
            )
            fit_time = time.time() - start_time

        return fit_time

    def predict(self, data, model, model_config, *, verbose=False, model_output_dir=None):
        self._validate_model_has_method(model, self._predict_method_name)

        model_predict_config = model_config.get(self._predict_method_name, {})
        model_predict_method = getattr(model, self._predict_method_name)

        start_timestamp = time.time()
        predictions, targets = model_predict_method(data, verbose=verbose, **model_predict_config)
        predict_time = time.time() - start_timestamp

        return predictions, targets, predict_time

    def fit_predict(
        self, train_data, model, model_config, *, refit_model=False, verbose=False, model_output_dir=None
    ):
        fit_time = self.fit(
            train_data, model, model_config, refit_model=refit_model, verbose=verbose, model_output_dir=model_output_dir
        )

        train_predictions, train_targets, predict_time = self.predict(
            train_data, model, model_config, verbose=verbose, model_output_dir=model_output_dir
        )

        return train_predictions, train_targets, fit_time, predict_time

    def score(self, predictions, targets):
        raise NotImplementedError()

    def plot(self, predictions, targets, scores, plot_dir: str):
        raise NotImplementedError()

    @staticmethod
    def _plot_base(predictions, actuals, plot_name: str, plot_directory: str, scores: dict, problem_name: str):
        if len(predictions) != len(actuals):
            raise ValueError('The length of the predictions must match the length of the actuals')

        # Create the title with the scores on it
        title = ProblemBase._make_plot_title('', scores)
        plt.title(title, fontsize=6)

        # Create the plot
        plt.xlabel('Predictions')
        plt.ylabel('Actuals')
        plt.scatter(predictions, actuals)
        plt.tight_layout()

        # Save the plot
        new_dir = os.path.join(plot_directory, problem_name)
        if not os.path.isdir(new_dir):
            os.makedirs(new_dir)
        file_name = os.path.join(new_dir, plot_name + '.pdf')
        plt.savefig(fname=file_name)
        plt.clf()

    @staticmethod
    def _make_plot_title(title, scores):
        for (score_name, score_value) in scores.items():
            if type(score_value) == dict:
                title += score_name.upper() + ':' + '\n'
                title = ProblemBase._make_plot_title(title, score_value)
            elif type(score_value) == np.float64 or type(score_value) == float:
                score_value = '{0:.5f}'.format(score_value)
                title += score_name + ': ' + score_value + '\n'
            else:
                title += score_name + ': ' + str(score_value) + '\n'
        return title


class RegressionProblem(ProblemBase):

    def __init__(self, group_key):
        super().__init__(group_key)
        self._predict_method_name = 'predict_regression'

    def score(self, predictions, targets):
        RMSEs = []
        pearson_losses = []
        pearson_coefs = []
        pearson_ps = []
        scores_by_group = {}

        for group, group_predictions in predictions.items():
            group_targets = targets[group]
            RMSE = rmse(group_predictions, group_targets)
            pearson_loss = pearson_loss_metric(group_predictions, group_targets)
            correlation, p_value = pearson_correlation(group_predictions, group_targets)

            scores_by_group[group] = {
                'rmse': RMSE,
                'pearson_loss': pearson_loss,
                'pearson_correlation': correlation,
                'pearson_p_value': p_value,
            }

            RMSEs.append(RMSE)
            pearson_losses.append(pearson_loss)
            pearson_coefs.append(correlation)
            pearson_ps.append(p_value)

        aggregate_scores = {
            'rmse_mean': np.mean(RMSEs),
            'rmse_std_dev': np.std(RMSEs, ddof=1),
            'pearson_loss_mean': np.mean(pearson_losses),
            'pearson_loss_std_dev': np.std(pearson_losses, ddof=1),
            'pearson_correlation_mean': np.mean(pearson_coefs),
            'pearson_correlation_mean_std_dev': np.std(pearson_coefs, ddof=1),
            'pearson_p_value_mean': np.mean(pearson_ps),
            'pearson_p_value_std_dev': np.std(pearson_ps, ddof=1),
        }

        return {
            self.group_scores_key: scores_by_group,
            self.agg_scores_key: aggregate_scores,
        }

    def plot(self, predictions, targets, scores, plot_dir: str):
        scores_by_group = scores[self.group_scores_key]

        # Plot per dataset
        for (group, group_predictions) in predictions.items():
            group_targets = targets[group]
            group_scores = scores_by_group[group]
            plot_name = group
            super()._plot_base(group_predictions, group_targets, plot_name, plot_dir, group_scores, type(self).__name__)


class RankProblem(ProblemBase):

    def __init__(self, group_key):
        super().__init__(group_key)
        self._predict_method_name = 'predict_rank'

    @staticmethod
    def _get_scores_by_group(rank_predictions, rank_problem_targets):
        scores_by_group = {}

        regression_targets = rank_problem_targets.regression_targets
        rank_targets = rank_problem_targets.rank_targets

        for group, group_regression_targets in regression_targets.items():
            group_rank_predictions = rank_predictions[group]
            group_rank_targets = rank_targets[group]

            scores_by_group[group] = {}

            correlation, p_value = spearman_correlation(group_rank_predictions, group_rank_targets)
            scores_by_group[group]['spearman_correlation'] = correlation
            scores_by_group[group]['spearman_p_value'] = p_value

            scores_by_group[group]['ndcg_at_k'] = ndcg_at_k(group_regression_targets, group_rank_predictions)
            scores_by_group[group]['regret_at_k'] = regret_at_k(group_regression_targets, group_rank_predictions)
            scores_by_group[group]['n_correct_at_k'] = n_correct_at_k(group_regression_targets, group_rank_predictions)

        return scores_by_group

    @staticmethod
    def _get_aggregate_scores(scores_by_group):
        aggregate_scores = {}
        total_scores = {}

        for score_name in ['spearman_correlation', 'spearman_p_value']:
            scores = list(group_score[score_name] for group, group_score in scores_by_group.items())
            aggregate_scores['{}_mean'.format(score_name)] = np.mean(scores)
            aggregate_scores['{}_std_dev'.format(score_name)] = np.std(scores, ddof=1)

        for score_name in ['ndcg_at_k', 'regret_at_k', 'n_correct_at_k']:
            scores = (group_score[score_name] for group, group_score in scores_by_group.items())
            scores_by_k = utils.transpose_jagged_2darray(scores)
            aggregate_scores['{}_mean'.format(score_name)] = [np.mean(scores_at_k) for i, scores_at_k in scores_by_k.items()]
            aggregate_scores['{}_std_dev'.format(score_name)] = [np.std(scores_at_k, ddof=1) for i, scores_at_k in scores_by_k.items()]

            flattened_scores = [score_at_k for i, scores_at_k in scores_by_k.items() for score_at_k in scores_at_k]
            total_scores['{}_mean'.format(score_name)] = np.mean(flattened_scores)
            total_scores['{}_std_dev'.format(score_name)] = np.std(flattened_scores, ddof=1)

        return aggregate_scores, total_scores

    def score(self, rank_predictions, rank_problem_targets):
        """Computes Spearman correlation, ndcg_at_k, regret_at_k, n_correct_at_k"""

        scores_by_group = self._get_scores_by_group(rank_predictions, rank_problem_targets)
        aggregate_scores, total_scores = self._get_aggregate_scores(scores_by_group)

        return {
            self.group_scores_key: scores_by_group,
            self.agg_scores_key: aggregate_scores,
            self.total_scores_key: total_scores
        }

    def plot(self, rank_predictions, rank_problem_targets, scores, plot_dir: str):
        scores_by_group = scores[self.group_scores_key]
        rank_targets = rank_problem_targets.rank_targets

        for dataset_id, predicted_ranks in rank_predictions.items():
            actual_ranks = rank_targets[dataset_id]
            group_scores = scores_by_group[dataset_id]
            group_scores = self.shorten_k_rank_scores(group_scores)
            plot_name = dataset_id + '_plot'
            super()._plot_base(predicted_ranks, actual_ranks, plot_name, plot_dir, group_scores, type(self).__name__)

    @staticmethod
    def shorten_k_rank_scores(rank_scores: dict):
        """Takes the average of the top k, k regret, and ndcg rank scores so the score at every k isn't reported"""

        rank_scores = rank_scores.copy()
        rank_scores['ndcg_at_k'] = np.mean(rank_scores['ndcg_at_k'])
        rank_scores['regret_at_k'] = np.mean(rank_scores['regret_at_k'])
        rank_scores['n_correct_at_k'] = np.mean(rank_scores['n_correct_at_k'])
        return rank_scores


def get_problem(problem_name: str, **kwargs):
    group_key = 'dataset_id'
    if problem_name == 'regression':
        return RegressionProblem(group_key)
    if problem_name == 'rank':
        return RankProblem(group_key)
