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


    def calculate_true_error(self, intervals: list):
        intersection_A = 0.0
        intersection_B = 0.0
        diff_A = 0.6
        diff_B = 0.4
        for interval in intervals:
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

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x_points = np.random.uniform(0, 1, m)
        y_points = np.array([self.generate_label(x) for x in x_points])
        result = np.column_stack((x_points, y_points))
        return result


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
        pass

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
        pass

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
        pass

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """
        # TODO: Implement me
        pass

    #################################
    # Place for additional methods


    #################################


if __name__ == '__main__':
    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

