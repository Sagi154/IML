#################################
# Your name:
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_points = np.random.uniform(0, 1, m)
        x_points = np.sort(x_points)
        y_points = np.array([self.generate_label(x) for x in x_points])
        samples = np.column_stack((x_points, y_points))
        return samples

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """
        steps_count = np.arange(m_first, m_last + 1, step)
        errors_arr = np.zeros((len(steps_count), 2))
        index = 0
        for n in steps_count:
            emp_error_sum = 0
            true_error_sum = 0
            for i in range(T):
                samples = self.sample_from_D(n)
                x_points = samples[:, 0]
                y_labels = samples[:, 1]
                inters, erm = intervals.find_best_interval(x_points, y_labels, k)
                emp_error_sum += erm / n
                true_error_sum += self.calculate_true_error(inters)
            errors_arr[index][0] = emp_error_sum / T
            errors_arr[index][1] = true_error_sum / T
            index += 1

        plt.title("experiment_m_range_erm")
        plt.xlabel('Samples Count')
        plt.ylabel('Errors Value')
        plt.plot(steps_count, errors_arr[:, 0], marker='o', linestyle='-', color='b', label='Average Empirical Error')
        plt.plot(steps_count, errors_arr[:, 1], marker='s', linestyle='--', color='g', label='Average True Error')
        plt.legend()
        plt.show()
        return errors_arr

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        samples = self.sample_from_D(m)
        x_points = samples[:, 0]
        y_labels = samples[:, 1]
        steps_count = np.arange(k_first, k_last + 1, step)
        errors_arr = np.zeros((len(steps_count), 2))
        index = 0
        optimal_k = k_first
        optimal_emp = np.inf
        for k in range(k_first, k_last + 1, step):
            inters, erm = intervals.find_best_interval(x_points, y_labels, k)
            empirical_error = erm / m
            if empirical_error < optimal_emp:
                optimal_k = k
                optimal_emp = empirical_error
            true_error = self.calculate_true_error(inters)
            errors_arr[index][0] = empirical_error
            errors_arr[index][1] = true_error
            index += 1
        plt.title("Experiment k range erm")
        plt.xlabel('k Count')
        plt.ylabel('Errors Value')
        plt.plot(steps_count, errors_arr[:, 0], marker='o', linestyle='-', color='b', label='Empirical Error')
        plt.plot(steps_count, errors_arr[:, 1], marker='s', linestyle='--', color='g', label='True Error')
        plt.legend()
        plt.show()
        print(f"Optimal k is: {optimal_k}")
        return optimal_k

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        samples = self.sample_from_D(m)
        x_points = samples[:, 0]
        y_labels = samples[:, 1]
        steps_count = np.arange(k_first, k_last + 1, step)
        errors_arr = np.zeros((len(steps_count), 4))
        index = 0
        optimal_k = k_first
        optimal_emp = np.inf
        for k in range(k_first, k_last + 1, step):
            inters, erm = intervals.find_best_interval(x_points, y_labels, k)
            empirical_error = erm / m
            true_error = self.calculate_true_error(inters)
            delta_k = 0.1 / np.power(k, 2)
            vcdim = 2 * k
            penalty = 2 * np.sqrt((vcdim + np.log(2 / delta_k)) / m)
            SRM = penalty + empirical_error
            if SRM < optimal_emp:
                optimal_k = k
                optimal_emp = SRM
            errors_arr[index][0] = empirical_error
            errors_arr[index][1] = true_error
            errors_arr[index][2] = penalty
            errors_arr[index][3] = SRM
            index += 1
        plt.title("experiment_k_range_srm")
        plt.xlabel('k Count')
        plt.ylabel('Errors Value')
        plt.plot(steps_count, errors_arr[:, 0], marker='o', linestyle='-', color='b', label='Empirical Error')
        plt.plot(steps_count, errors_arr[:, 1], marker='s', linestyle='--', color='g', label='True Error')
        plt.plot(steps_count, errors_arr[:, 2], marker='o', linestyle='--', color='orange', label='Penalty')
        plt.plot(steps_count, errors_arr[:, 3], marker='o', linestyle='--', color='y', label='Penalty + Empirical Error')
        plt.legend()
        plt.show()
        print(f"optimal k for d:{optimal_k}")
        return optimal_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        samples = self.sample_from_D(m)
        np.random.shuffle(samples)
        split_index = int(0.8 * m)
        train_samples = samples[:split_index]
        train_samples = train_samples[train_samples[:, 0].argsort()]
        x_points = train_samples[:, 0]
        y_labels = train_samples[:, 1]
        holdout_samples = samples[split_index:]
        holdout_samples = holdout_samples[holdout_samples[:, 0].argsort()]
        best_inters = []
        emp_error_holdout_samples = np.empty(10)
        for k in range(1, 11):
            inters, erm = intervals.find_best_interval(x_points, y_labels, k)
            best_inters.append(inters)

        for i, best_inter in enumerate(best_inters):
            emp_error_holdout_samples[i] = self.calculate_empirical_error(best_inter, holdout_samples)

        optimal_k = np.argmin(emp_error_holdout_samples) + 1
        print(f"optimal k for e:{optimal_k}")
        return optimal_k


    #################################
    # Place for additional methods
    def calculate_true_error(self, inters: list):
        intersection_A = 0.0
        intersection_B = 0.0
        diff_A = 0.6
        diff_B = 0.4
        for interval in inters:
            # Calculate A and I intersection, B and I intersection
            intersection_A += max(0, (min(interval[1], 0.2) - interval[0]))
            intersection_A += max(0, (min(interval[1], 0.6) - max(interval[0], 0.4)))
            intersection_A += max(0, (interval[1] - max(interval[0], 0.8)))
            intersection_B += max(0, (min(interval[1], 0.4) - max(interval[0], 0.2)))
            intersection_B += max(0, (min(interval[1], 0.8) - max(interval[0], 0.6)))
        # Calculate A minus I, B minus I
        diff_A -= intersection_A
        diff_B -= intersection_B
        return 0.2 * intersection_A + 0.9 * intersection_B + 0.8 * diff_A + 0.1 * diff_B

    def generate_label(self, x):
        if 0 <= x <= 0.2 or 0.4 <= x <= 0.6 or 0.8 <= x <= 1:
            P_y1 = 0.8  # P[y=1|x] = 0.8
        elif 0.2 < x < 0.4 or 0.6 < x < 0.8:
            P_y1 = 0.1  # P[y=1|x] = 0.1
        y = np.random.rand() < P_y1
        return int(y)

    def calculate_empirical_error(self, inters, samples):
        sample_size = len(samples)
        error_count = 0
        for x, y in samples:
            expected = 0
            for interval in inters:
                left = interval[0]
                right = interval[1]
                if left <= x <= right:
                    expected = 1
            error_count += (y != expected)
        return error_count / sample_size


    #################################


if __name__ == '__main__':
    ass = Assignment2()
    # ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

