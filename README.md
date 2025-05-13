# TNO Quantum: Optimization - QUBO - Preprocessors

TNO Quantum provides generic software components aimed at facilitating the development
of quantum applications.

This package contains implementations of QUBO preprocessors.

## Documentation

Documentation of the `tno.quantum.optimization.qubo.preprocessors` package can be found [here](https://tno-quantum.github.io/documentation/).


## Install

Easily install the `tno.quantum.optimization.qubo.preprocessors` package using pip:

```console
$ python -m pip install tno.quantum.optimization.qubo.preprocessors
```

## Usage

The following example shows how to list the available preprocessor and how to instantiate them.

```python
from tno.quantum.optimization.qubo.components import PreprocessorConfig
supported_preprocessors = PreprocessorConfig.supported_items()
preprocessor = PreprocessorConfig(name='q_pro_plus_preprocessor').get_instance()
```

Alternatively, a preprocessor can also be instantiated directly.

```python
from tno.quantum.optimization.qubo.preprocessors import QProPlusPreprocessor
preprocessor = QProPlusPreprocessor(max_iter=10)
```

## (End)use limitations
The content of this software may solely be used for applications that comply with international export control laws.
