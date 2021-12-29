"""
Copyright (C) 2021  Patrick Schwab, GlaxoSmithKline plc, Sonali Parbhoo, Harvard University
Copyright (C) 2019  Patrick Schwab, ETH Zurich

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions
 of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
import os
import numpy as np
from ncore.apps.util import convert_int_to_indicator_list


def plot_treatment_effect_distributions(potential_outcomes, output_directory, max_num_plots=16,
                                        file_name="treatment_effect_distributions.pdf"):
    import seaborn as sns
    import matplotlib.pyplot as plt

    num_simulated_treatments = int(np.log2(potential_outcomes.shape[-1]))
    for offset in list(range(2 ** num_simulated_treatments))[:max_num_plots]:
        if offset == 0:
            # Skip no treatment applied.
            continue
        treatment_label = ''.join(map(str, convert_int_to_indicator_list(offset,
                                                                         min_length=num_simulated_treatments)))
        sns.distplot(potential_outcomes[:, offset], hist=False, kde=True,
                     kde_kws={'shade': True, 'linewidth': 3},
                     label=treatment_label)

    plt.legend(prop={'size': 16}, title='Treatment(s)')
    plt.title('Treatment effects by treatment / combination of treatments')
    plt.xlabel('Treatment effect')
    plt.ylabel('Density')
    plt.savefig(os.path.join(output_directory, file_name))
    plt.close("all")
