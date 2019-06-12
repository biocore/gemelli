[![Build Status](https://travis-ci.org/cameronmartino/gemelli.svg?branch=master)](https://travis-ci.org/cameronmartino/gemelli)
[![Coverage Status](https://coveralls.io/repos/github/cameronmartino/gemelli/badge.svg?branch=master)](https://coveralls.io/github/cameronmartino/gemelli?branch=master)

**gemelli is still being developed, so backwards-incompatible changes might occur.**
**If you have any questions, feel free to contact the development team at cmartino at eng.ucsd.edu.**

# gemelli

## usage

```python
import numpy as np
import pandas as pd
from gemelli.factorization import TenAls
from gemelli.preprocessing import build, rclr

# contruct and transform the tensor
tensor = Build()
tensor.construct(table, metadata, subjects,
                 [condition_1, condition_2, ..., condition_n])
tensor_rclr = rclr(tensor.counts)
# factorize
TF = TenAls().fit(tensor_rclr)
# write loading files 
PC = ['PC'+str(i+1) for i in range(rank)]
# loadings as daaframe
sample_loading = pd.DataFrame(abs(TF.sample_loading),
                              tensor.subject_order)
feature_loading = pd.DataFrame(TF.feature_loading,
                               tensor.feature_order)
temporal_loading = pd.DataFrame(TF.conditional_loading,
                                tensor.condition_orders[0])
```

## resources

Named after gemelli by alighiero boetti and also the pasta. 

[TenAls translated from Sewoong Oh](http://swoh.web.engr.illinois.edu/software/optspace/code.html)
