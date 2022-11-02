# Python implementation for "Adapting to Online Label Shift with Provable Guarantees"

This is the python implementation for the experiments in the paper "**Adapting to Online Label Shift with Provable Guarantees**" [1]. We will first introduce the structure and requirements of the code, followed by a brief instruction for a quick start.

## Code Structure
`main.py` is the entrance of the program. The code mainly contains the following three parts.

* **online**: Implementations of the proposed algorithms (ATLAS and ATLAS-ADA).
  * `models/ogd.py` implements the base-algorithm.
  * `models/meta.py` implements the meta-algorithm.
  * `models/atlas.py` implements the ATLAS and ATLAS-ADA algorithms.
  * the subfolder *utils* contains the codes for the basic components of ATLAS and ATLAS-ADA, including the implementations of the unbiased risk estimator (`risk.py`), step size tuning (`schedule.py`,`lr.py`) and the hint function `hint_function.py`, etc.

* **dataset**: Code for datasets pre-processing.
  * `toy_data.py` is the code for the simulated experiments on benchmark datasets.
  * `table_data.py` is the code for the real-life SHL dataset for the locomotion detection task.

* **utils**: Useful tools for running the code.
  * `argparse.py` parses the configurations.
  * `multi_thread.py` is a tool multi-threaded acceleration.
  * `shift_simulate.py` is a tool to generate the simulated label shifts.
  * `tools.py` includes other useful tools, including a timer for testing the program runtime. 


## Requirements
* matplotlib==3.3.3
* numpy==1.19.2
* pandas==1.4.1
* prettytable==0.9.2
* PyYAML==6.0
* scikit_learn==0.23.2
* torch==1.7.0
* tqdm==4.62.3

## Quick Start

We provide several demos in the folder *./demos*. To get a quick start, the readers can run any demo by `bash train.sh` in the corresponding path with the parameter specified in `config.yaml`. 

For example, one can use ``.\demos\locomotion\ATLAS\train.sh`` to run ATLAS algorithm over the SHL datasets [2], whose parameters are set as ``.\demos\locomotion\ATLAS\config.yaml``

## Reference:
[1] Yong Bai\*, Yu-Jie Zhang\*, Peng Zhao, Masashi Sugiyama, and Zhi-Hua Zhou. Adapting to Online Label Shift with Provable Guarantees. In: Advances in Neural Information Processing Systems 35 (NeurIPS 2022), New Orleans, Louisiana, 2022. 



