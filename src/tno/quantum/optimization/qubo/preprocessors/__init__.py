"""This package contains implementations of QUBO :py:class:`~tno.quantum.optimization.qubo.components.Preprocessor` s.

Example:
-------
The following example shows how to list the available preprocessors and how to instantiate them.

>>> from tno.quantum.optimization.qubo.components import PreprocessorConfig
>>> list(PreprocessorConfig.supported_items())
['q_pro_plus_preprocessor']
>>> preprocessor = PreprocessorConfig(name='q_pro_plus_preprocessor').get_instance()

Alternatively, a preprocessor can also be instantiated directly.

>>> from tno.quantum.optimization.qubo.preprocessors import QProPlusPreprocessor
>>> preprocessor = QProPlusPreprocessor(max_iter=10)

Once a preprocessor is instantiated, it can be used to preprocess a QUBO as follows.

>>> from tno.quantum.optimization.qubo.components import QUBO
>>> # Construct QUBO
>>> qubo = QUBO([
...     [ 7,  6, -2,  9,  5],
...     [ 8, -5,  9, -7, -5],
...     [ 7,  9, -6, -3, -4],
...     [ 7, -8,  1,  8,  6],
...     [ 1,  1,  7,  8, -2]
... ])

>>> # Preprocess QUBO
>>> partial_solution, qubo_reduced = preprocessor.preprocess(qubo)
>>> partial_solution
PartialSolution {x_0 = 0, x_1 = 1 - x_2}
>>> qubo_reduced
QUBO of dim: 3x3, with 9 non zero elements.

>>> # Once the solution of the reduced QUBO is found, it can be expanded to the
>>> # solution of the original QUBO.
>>> solution_reduced = [0, 1, 0]
>>> solution = partial_solution.expand(solution_reduced)
>>> solution
BitVector(01010)
"""  # noqa: E501

from tno.quantum.optimization.qubo.preprocessors._qpro_plus import QProPlusPreprocessor

__all__ = ["QProPlusPreprocessor"]

__version__ = "1.0.0"
