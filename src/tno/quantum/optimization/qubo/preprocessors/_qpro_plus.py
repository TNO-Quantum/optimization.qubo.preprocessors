"""This module contains the ``QProPlusPreprocessor`` class."""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from tno.quantum.optimization.qubo.components import QUBO, PartialSolution, Preprocessor


class QProPlusPreprocessor(Preprocessor):
    """Preprocessor class that partially solves QUBOs using the QPro+ algorithm.

    The :py:class:`QProPlusPreprocessor` is a preprocessor that applies the `QPro+
    algorithm` to reduce the size of a QUBO, while maintaining an optimal solution, as
    described in: `'Logical and inequality implications for reducing the size and
    difficulty of QUBO problems'` by Fred Glover, Mark Lewis, Gary Kochenberger
    (https://www.sciencedirect.com/science/article/pii/S0377221717307567).

    .. note::
        In our implementation, the reduction rules are applied with the aim to
        minimize the QUBO objective, while the convention of the paper is to maximize
        the QUBO objective.

    Example:
        >>> from tno.quantum.optimization.qubo.components import QUBO
        >>> from tno.quantum.optimization.qubo.preprocessors import QProPlusPreprocessor
        >>>
        >>> # Create an example QUBO
        >>> qubo = QUBO([
        ...     [ 7,  6, -2,  9,  5],
        ...     [ 8, -5,  9, -7, -5],
        ...     [ 7,  9, -6, -3, -4],
        ...     [ 7, -8,  1,  8,  6],
        ...     [ 1,  1,  7,  8, -2]
        ... ])
        >>>
        >>> # Preprocess QUBO using QProPlusPreprocessor
        >>> preprocessor = QProPlusPreprocessor(max_iter=10)
        >>> partial_solution, qubo_reduced = preprocessor.preprocess(qubo)
        >>> partial_solution
        PartialSolution {x_0 = 0, x_1 = 1 - x_2}
        >>> qubo_reduced
        QUBO of dim: 3x3, with 9 non zero elements.
        >>>
        >>> # Once the solution of the reduced QUBO is found, it can be expanded to the
        >>> # solution of the original QUBO.
        >>> solution_reduced = [0, 1, 0]
        >>> solution = partial_solution.expand(solution_reduced)
        >>> solution
        BitVector(01010)

    Attributes:
        is_converged: If ``True``, indicates that no more rules can be applied.
    """

    def __init__(self, *, max_iter: int | None = None, verbose: bool = False) -> None:
        """Init :py:class:`QProPlusPreprocessor`.

        Args:
            max_iter: Maximum number of iterations of applying the rules.
                One iteration reduces the size of the QUBO by at most one. If not
                provided, it will be set to the size of the QUBO.
            verbose:
                verbose: If ``True``, the rules which are being applied are printed.
        """
        # Flag to keep track of whether the algorithm has converged
        self.is_converged = False

        # Flag to keep
        self.verbose = verbose

        # Since every (effective) iteration removes one variable, no more than qubo.size
        # iterations are required
        self._max_iter = max_iter

        # Create logger
        self._logger = logging.getLogger("QProPlus")
        self._logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        self._logger.addHandler(handler)

    def preprocess(self, qubo: QUBO) -> tuple[PartialSolution, QUBO]:
        """Performs the reduction algorithm for a maximum number of iterations.

        Args:
            qubo: QUBO to be preprocessed.

        Returns:
            Partial solution and corresponding preprocessed QUBO.
        """
        # Get matrix and offset from QUBO, adding a minus sign to switch max/min
        self._Q = -(qubo.matrix + qubo.matrix.T) / 2
        self._c0 = -qubo.offset

        # Store values for variables which are solved for in a PartialSolution
        # Keep track of the (indices of the) free variables in a list
        self._partial_solution = PartialSolution(qubo.size)
        self._free_variables = list(range(qubo.size))

        # Compute D_i^+ and D_i^-
        self._compute_D_positive_negative()

        # Flag to keep track of whether the algorithm has converged
        self.is_converged = False

        # Since every (effective) iteration removes one variable, no more than qubo.size
        # iterations are required
        max_iter = self._max_iter or qubo.size

        for _it in range(max_iter):
            if self.is_converged:
                break

            if self.verbose:
                self._logger.info("Iteration %d:", _it)
                self._logger.info(
                    "  Remaining variables: %s", str(self._free_variables)
                )

            self._do_iteration()

        # Construct QUBO object from the resulting Q and c0.
        # Again, minus signs are added to switch max/min
        qubo = QUBO(-self._Q, -self._c0)
        return self._partial_solution, qubo

    def _do_iteration(self) -> None:
        """Perform a single iteration of the algorithm.

        A single iteration can remove at most one variable.
        """
        # i: index of variable w.r.t. self.Q
        # i_orig: index of variable in terms of the original QUBO
        for i, i_orig in enumerate(self._free_variables):
            if self.verbose:
                self._logger.info("  Considering variable %d", i_orig)

            # Apply rule 1.0 to x_i
            if self._check_rule_1(i):  # x_i = 1
                self._apply_rule_1(i)
                return

            # Apply rule 2.0 to x_i
            if self._check_rule_2(i):  # x_i = 0
                self._apply_rule_2(i)
                return

            # Apply rules to x_i, x_h
            for h, h_orig in enumerate(self._free_variables[:i]):
                if self.verbose:
                    self._logger.info(
                        "  Considering variables %i and %d", i_orig, h_orig
                    )

                # Apply rules 3.1 to 3.4
                outcome_rule_3 = self._check_rule_3(i, h)
                if outcome_rule_3:
                    [to_set_i, to_set_h] = outcome_rule_3
                    self._apply_rule_3(i, h, to_set_i, to_set_h)
                    return

                # Apply rules 2.5 and 2.6
                outcome_rule_2 = self._check_rule_2_extended(i, h)
                if outcome_rule_2 == 1:  # 2.5: x_i + x_h = 1
                    self._apply_rule_2_5(i, h)
                    return

                if outcome_rule_2 == 2:  # 2.6: x_i = x_h
                    self._apply_rule_2_6(i, h)
                    return

        self.is_converged = True

    def _compute_D_positive_negative(self) -> None:  # noqa: N802
        """Computes D_i^+ and D_i^-."""
        # NOTE: This method can possibly be optimized
        off_diagonal = self._Q.copy()
        np.fill_diagonal(off_diagonal, 0)  # Set diagonal elements to 0

        positive_off_diagonal_elements = np.where(off_diagonal > 0, off_diagonal, 0)
        negative_off_diagonal_elements = np.where(off_diagonal < 0, off_diagonal, 0)

        self.D_plus = np.sum(positive_off_diagonal_elements, axis=0) + np.sum(
            positive_off_diagonal_elements, axis=1
        )
        self.D_minus = np.sum(negative_off_diagonal_elements, axis=0) + np.sum(
            negative_off_diagonal_elements, axis=1
        )

    def _update_set_variable_value(self, i: int, value: int) -> None:
        r"""Update to solve for x_i := value, removing the variable x_i.

        c_j := c_j + d_ij for all j ∈ N \ {i}
        c_0 := c_0 + c_i
        N := N \ {i}
        """
        d_i = self._Q[i] + self._Q[:, i]
        if value == 1:
            self._c0 += self._Q[i, i]
            # add d_i to the diagonal
            np.fill_diagonal(self._Q, self._Q.diagonal() + d_i)

        positive_idx = np.where(d_i > 0)
        negative_idx = np.where(d_i < 0)
        self.D_plus[positive_idx] -= d_i[positive_idx]
        self.D_minus[negative_idx] -= d_i[negative_idx]

        self.D_minus = np.delete(self.D_minus, i)
        self.D_plus = np.delete(self.D_plus, i)
        self._Q = np.delete(np.delete(self._Q, i, 0), i, 1)

        # Remove variable i and update partial solution
        i_orig = self._free_variables.pop(i)  # N := N \ {i}
        self._partial_solution.assign_value(i_orig, value)

    def _update_set_variable_equal(self, i: int, h: int) -> None:
        r"""Update to solve for x_h := x_i, removing the variable x_h.

        c_i := c_i + c_h + d_ih
        d_ij := d_ij + d_hj for all j ∈ N \ {i,h}
            Q_ij := Q_ij + Q_hj
            Q_ji := Q_ji + Q_jh
            d_ij := Q_ij + Q_ji = Q_ij + Q_ji + Q_hj + Q_jh = d_ij - d_hj
        N := N \ {h}
        """
        self._Q[i] += self._Q[h]
        self._Q[:, i] += self._Q[:, h]
        self._Q = np.delete(np.delete(self._Q, h, 0), h, 1)

        # Recalculate D_i^+ and D_i^-
        self._compute_D_positive_negative()

        # Remove variable h and update partial solution
        i_orig = self._free_variables[i]
        h_orig = self._free_variables.pop(h)  # N := N \ {h}
        self._partial_solution.assign_variable(h_orig, i_orig, conj=False)

    def _update_set_variable_unequal(self, i: int, h: int) -> None:
        r"""Update to solve for x_h := 1 - x_i, removing the variable x_h.

        c_i := c_i - c_h
        c_0 := c_0 + c_h
        c_j := c_j + d_jh for all j ∈ N \ {i, h}
        d_ij := d_ij - d_hj for all j ∈ N \ {i, h}
            Q_ij := Q_ij - Q_hj
            Q_ji := Q_ji - Q_jh
            d_ij := Q_ij + Q_ji = Q_ij + Q_ji - Q_hj - Q_jh = d_ij - d_hj
        N := N \ {h}
        """
        self._c0 += self._Q[h, h]
        self._Q[i] -= self._Q[h]
        self._Q[:, i] -= self._Q[:, h]
        np.fill_diagonal(self._Q, self._Q.diagonal() + self._Q[h] + self._Q[:, h])
        self._Q = np.delete(np.delete(self._Q, h, 0), h, 1)

        # Recalculate D_i^+ and D_i^-
        self._compute_D_positive_negative()

        # Remove variable h and update partial solution
        i_orig = self._free_variables[i]
        h_orig = self._free_variables.pop(h)  # N := N \ {h}
        self._partial_solution.assign_variable(h_orig, i_orig, conj=True)

    def _check_rule_1(self, i: int) -> bool:
        """Check the conditions for rule 1.0.

        Rule 1.0:

            If c_i + D_i^- >= 0, then x_i = 1 is optimal.
        """
        c_i: float = self._Q[i, i]
        D_i_negative: float = self.D_minus[i]  # noqa: N806

        return c_i + D_i_negative >= 0

    def _check_rule_2(self, i: int) -> bool:
        """Check the conditions for rule 2.0.

        Rule 2.0:

            If c_i + D_i^+ <= 0, then x_i = 0 is optimal.
        """
        c_i: float = self._Q[i, i]
        D_i_positive: float = self.D_plus[i]  # noqa: N806

        return c_i + D_i_positive <= 0

    def _check_rule_3(self, i: int, h: int) -> Literal[False] | tuple[int, int]:
        """Check the conditions for rules 3.1-3.4.

        Rule 3.1 (d_{ih} >= 0):

            If c_i + c_h - d_{ih} + D_i^+ + D_h^+ <= 0, then x_i = x_h = 0 in an
            optimal solution.

        Rule 3.2 (d_{ih} < 0):

            If -c_i + c_h + d_{ih} - D_i^- + D_h^+ <= 0, then x_i = 1 and x_h=0 in an
            optimal solution.

        Rule 3.3 (d_{ih} < 0):

            If c_i - c_h + d_{ih} + D_i^+ - D_h^- <= 0, then x_i = 0 and x_h = 1 in an
            optimal solution.

        Rule 3.4 (d_{ih} >= 0):

            If -c_i - c_h - d_{ih} - D_i^- - D_h^- >= 0, then x_i = x_h = 1 in an
            optimal solution.
        """
        c_i, c_h = self._Q[i, i], self._Q[h, h]
        D_i_positive, D_h_positive = self.D_plus[i], self.D_plus[h]  # noqa: N806
        D_i_negative, D_h_negative = self.D_minus[i], self.D_minus[h]  # noqa: N806
        d_ih = self._Q[i, h] + self._Q[h, i]

        # Rule 3.1
        if d_ih >= 0 and c_i + c_h - d_ih + D_i_positive + D_h_positive <= 0:
            i, h = 0, 0
            return i, h

        # Rule 3.2
        if d_ih < 0 and -c_i + c_h + d_ih - D_i_negative + D_h_positive <= 0:
            i, h = 1, 0
            return i, h

        # Rule 3.3
        if d_ih < 0 and c_i - c_h + d_ih + D_i_positive - D_h_negative <= 0:
            i, h = 0, 1
            return i, h

        # Rule 3.4
        if d_ih >= 0 and -c_i - c_h - d_ih - D_i_negative - D_h_negative <= 0:
            i, h = 1, 1
            return i, h

        return False

    def _check_rule_2_extended(self, i: int, h: int) -> Literal[False] | int:
        """Check the conditions for rules 2.5 and 2.6.

        Rule 2.5 (d_{ih} < 0):

            If c_i - d_{ih} + D_i^- >= 0 and c_i + d_{ih} + D_i^+ <= 0 (or swap i, h),
            then x_i + x_h = 1 in an optimal solution.

        Rule 2.6 (d_{ih} > 0):

            If c_i - d_{ih} + D_i^+ <= 0 and c_i + d_{ih} + D_i^- >= 0 (or swap i, h),
            then x_i = x_h in an optimal solution.
        """
        c_i, c_h = self._Q[i, i], self._Q[h, h]
        D_i_positive, D_h_positive = self.D_plus[i], self.D_plus[h]  # noqa: N806
        D_i_negative, D_h_negative = self.D_minus[i], self.D_minus[h]  # noqa: N806
        d_ih = self._Q[i, h] + self._Q[h, i]

        if d_ih < 0:
            a1 = c_i - d_ih + D_i_negative >= 0
            a2 = c_h - d_ih + D_h_negative >= 0
            b1 = c_i + d_ih + D_i_positive <= 0
            b2 = c_h + d_ih + D_h_positive <= 0
            if (a1 or a2) and (b1 or b2):
                # Optimal solution has x_i + x_h = 1
                return 1
        elif d_ih > 0:
            c1 = c_i - d_ih + D_i_positive <= 0
            c2 = c_h + d_ih + D_h_negative >= 0
            d1 = c_i + d_ih + D_i_negative >= 0
            d2 = c_h - d_ih + D_h_positive <= 0
            if (c1 or c2) and (d1 or d2):
                # Optimal solution has x_i = x_h
                return 2
        return False

    def _apply_rule_1(self, i: int) -> None:
        """Apply rule 1."""
        if self.verbose:
            i_orig = self._free_variables[i]
            msg = f"    Rule 1.0 applies for variable {i_orig} (current index {i}) as "
            msg += f"{self._Q[i, i] + self.D_minus[i]} >= 0"
            self._logger.info(msg)
        self._update_set_variable_value(i, 1)

    def _apply_rule_2(self, i: int) -> None:
        """Apply rule 2."""
        if self.verbose:
            i_orig = self._free_variables[i]
            msg = f"    Rule 2.0 applies for variable {i_orig} (current index {i}) as "
            msg += f"{self._Q[i, i] + self.D_plus[i]} <= 0"
            self._logger.info(msg)
        self._update_set_variable_value(i, 0)

    def _apply_rule_3(self, i: int, h: int, to_set_i: int, to_set_h: int) -> None:
        """Apply rule 3."""
        h_orig = self._free_variables[h]
        if self.verbose:
            rule_idx = 1 + to_set_i + 2 * to_set_h
            i_orig = self._free_variables[i]
            msg = f"    Rule 3.{rule_idx} applies for variables {i_orig} and {h_orig}:"
            msg += f" set x_{i_orig} = {to_set_i} and x_{h_orig} = {to_set_h}"
            self._logger.info(msg)
        self._update_set_variable_value(i, to_set_i)
        # Note: index h may have changed by deleting variable i
        h = self._free_variables.index(h_orig)
        self._update_set_variable_value(h, to_set_h)

    def _apply_rule_2_5(self, i: int, h: int) -> None:
        """Apply rule 2.5."""
        i_orig = self._free_variables[i]
        h_orig = self._free_variables[h]
        if self.verbose:
            self._logger.info(
                "    Rule 2.5 applies for variables %d and %d", i_orig, h_orig
            )
        self._update_set_variable_unequal(i, h)

    def _apply_rule_2_6(self, i: int, h: int) -> None:
        """Apply rule 2.6."""
        i_orig = self._free_variables[i]
        h_orig = self._free_variables[h]
        if self.verbose:
            self._logger.info(
                "    Rule 2.6 applies for variables %d and %d", i_orig, h_orig
            )
        self._update_set_variable_equal(i, h)


def reduce_qpro_plus(
    qubo: QUBO, max_iter: int | None = None, *, verbose: bool = False
) -> tuple[PartialSolution, QUBO]:
    """Reduce dimensionality of QUBO, using the ``QProPlusPreprocessor``.

    A wrapper function that constructs a ``QProPlusPreprocessor`` and uses it to reduce
    the QUBO.

    Args:
        qubo: The QUBO to be reduced.
        max_iter: Maximum number of iterations of applying the rules.
            One iteration reduces the size of the QUBO by at most one. If not provided,
            it will be set to the size of the QUBO.
        verbose: If set to True, the rules which are being applied are logged (debug
            level).

    Returns:
        A partial solution object and the corresponding lower-dimensional QUBO.
    """
    if max_iter is None:
        max_iter = qubo.size

    preprocessor = QProPlusPreprocessor(max_iter=max_iter, verbose=verbose)
    return preprocessor.preprocess(qubo)
