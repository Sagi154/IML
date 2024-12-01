#################################
# Your name: Roei Lahav
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
        # TODO: Implement me
        x = np.random.uniform(0, 1, m)
        sorted_x = np.sort(x)
        y = np.zeros(m)
        for i in range(m):
            if (0 <= sorted_x[i] <= 0.2) or (0.4 <= sorted_x[i] <= 0.6) or (0.8 <= sorted_x[i] <= 1):
                y[i] = 1 if np.random.rand() < 0.8 else 0
            else:
                y[i] = 1 if np.random.rand() < 0.1 else 0

        samples = np.column_stack((sorted_x, y))
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
        # TODO: Implement the loop
        n_steps = np.arange(m_first, m_last + 1, step)
        errors_array = np.zeros((len(n_steps), 2))

        index = 0

        for n in n_steps:
            emp_error_sum = 0
            true_error_sum = 0
            for i in range(T):
                samples = ass.sample_from_D(n)
                x = samples[:, 0]
                y = samples[:, 1]
                best_interval, err_count = intervals.find_best_interval(x, y, k)
                emp_error_sum += err_count / n
                true_error_sum += self.calculate_true_error(best_interval)

            errors_array[index][0] = emp_error_sum / T
            errors_array[index][1] = true_error_sum / T
            index += 1

        plt.title("experiment_m_range_erm")
        plt.xlabel('Samples')
        plt.ylabel('Errors')
        plt.plot(n_steps, errors_array[:, 0], marker='o', linestyle='-', color='b', label='Average Empirical Error')
        plt.plot(n_steps, errors_array[:, 1], marker='s', linestyle='--', color='r', label='Average True Error')
        plt.legend()
        plt.show()

        return errors_array

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        # TODO: Implement the loop
        samples = self.sample_from_D(m)
        x = samples[:, 0]
        y = samples[:, 1]

        k_steps = np.arange(k_first, k_last + 1, step)
        errors_array = np.zeros((len(k_steps), 2))

        best_k = k_first
        min_emp_error = 0
        flag = True
        index = 0

        for k in k_steps:
            best_interval, err_count = intervals.find_best_interval(x, y, k)
            emp_error = err_count / m
            true_error = self.calculate_true_error(best_interval)
            if flag:
                min_emp_error = emp_error
                flag = False
            if emp_error < min_emp_error:
                best_k = k
                min_emp_error = emp_error

            errors_array[index][0] = emp_error
            errors_array[index][1] = true_error
            index += 1

        plt.title("Experiment k range erm")
        plt.xlabel("k")
        plt.plot(k_steps, errors_array[:, 0], marker='o', label="empirical error")
        plt.plot(k_steps, errors_array[:, 1], marker='o', label="true error")
        plt.legend()
        plt.show()
        return best_k

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
        # TODO: Implement the loop
        samples = self.sample_from_D(m)
        x = samples[:, 0]
        y = samples[:, 1]

        k_steps = np.arange(k_first, k_last + 1, step)
        errors_array = np.zeros((len(k_steps), 4))

        best_k = k_first
        min_SRM = -1
        flag = True
        index = 0

        for k in k_steps:
            best_interval, err_count = intervals.find_best_interval(x, y, k)
            emp_error = err_count / m
            true_error = self.calculate_true_error(best_interval)
            penalty = 2 * np.sqrt((2 * k + np.log(2 * np.power(k, 2) / 0.1)) / m)
            SRM = emp_error + penalty
            if flag:
                min_SRM = SRM
                flag = False
            if SRM < min_SRM:
                best_k = k
                min_SRM = SRM

            errors_array[index][0] = emp_error
            errors_array[index][1] = true_error
            errors_array[index][2] = penalty
            errors_array[index][3] = SRM

            index += 1

        plt.title("Experiment k range srm")
        plt.xlabel("k")
        plt.plot(k_steps, errors_array[:, 0], marker='o', label="empirical error")
        plt.plot(k_steps, errors_array[:, 1], marker='o', label="true error")
        plt.plot(k_steps, errors_array[:, 2], marker='o', label="penalty")
        plt.plot(k_steps,errors_array[:, 3], marker='o', label="penalty + empirical error")
        plt.legend()
        plt.show()
        return best_k

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me

        best_intervals = []
        emp_errors_s2 = np.empty(10)

        samples = self.sample_from_D(m)
        np.random.shuffle(samples)
        s1, s2 = samples[:int(m * 0.8)], samples[int(m * 0.8):]
        s1 = s1[s1[:, 0].argsort()]

        x = s1[:, 0]
        y = s1[:, 1]

        for k in range(1, 11):
            best_interval, err_count = intervals.find_best_interval(x, y, k)
            best_intervals.append(best_interval)

        index = 0
        for best_interval in best_intervals:
            emp_errors_s2[index] = self.calculate_empirical_error(s2,best_interval)
            index += 1

        return np.argmin(emp_errors_s2) + 1

    #################################
    # Place for additional methods
    def calculate_true_error(self, I):
        J = [(0, 0.2), (0.4, 0.6), (0.8, 1)]
        JComplements = [(0.2, 0.4), (0.6, 0.8)]
        I_intersection_J = self.calculate_intersection(I, J)
        I_intersection_JComplements = self.calculate_intersection(I, JComplements)
        IComplements_intersection_J = 0.6 - I_intersection_J
        IComplements_intersection_JComplements = 0.4 - I_intersection_JComplements
        return 0.8 * IComplements_intersection_J + 0.1 * IComplements_intersection_JComplements + 0.2 * I_intersection_J + 0.9 * I_intersection_JComplements

    def calculate_intersection(self, I, J):
        len_intersection = 0
        point1 = 0
        point2 = 0
        while point1 < len(I) and point2 < len(J):
            start = max(I[point1][0], J[point2][0])
            end = min(I[point1][1], J[point2][1])
            if start < end:
                len_intersection += end - start
            if I[point1][1] == J[point2][1]:
                point1 += 1
                point2 += 1
            elif I[point1][1] < J[point2][1]:
                point1 += 1
            else:
                point2 += 1
        return len_intersection

    def calculate_empirical_error(self, samples, hypothesis):
        n = len(samples)
        errors_count = 0
        for x, y in samples:
            expected = 0
            for interval in hypothesis:
                a = interval[0]
                b = interval[1]
                if a <= x <= b:
                    expected = 1
            if expected != y:
                errors_count += 1
        return errors_count / n

    #################################


if __name__ == '__main__':

    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    # ass.experiment_k_range_erm(1500, 1, 10, 1)
    # ass.experiment_k_range_srm(1500, 1, 10, 1)
    # ass.cross_validation(1500)

