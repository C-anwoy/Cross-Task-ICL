# Language Models can Exploit Cross-Task In-context Learning for Data-Scarce Novel Tasks
Code corresponding to the paper <a href="https://arxiv.org/abs/2405.10548" target="_blank">Language Models can Exploit Cross-Task In-context Learning for Data-Scarce Novel Tasks</a> accepted to the Main Conference of ACL'24.

![alt text](https://github.com/C-anwoy/Cross-Task-ICL/blob/main/images/Cross-task-ICL-1.png?raw=true)

## Replication Instructions

### Dataset creation
To generate the dataset splits of taget, source and psudo-target. First run the `data_preprocess.ipynb file`. This will generate the various splits in the data folder.

Then extract all the similar prompts by running the `get_similar_prompts.py` script. This will generate target-prompts and target-psudo-prompts in the data folder. Note pass `--make_psudo` argument to generate target-psudo-prompts.

### Running main results
Use the drive folder for scripts to reproduce main results.

## Paper Citation
If you found the paper helpful, consider citing it:
```
@misc{chatterjee2024language,
      title={Language Models can Exploit Cross-Task In-context Learning for Data-Scarce Novel Tasks}, 
      author={Anwoy Chatterjee and Eshaan Tanwar and Subhabrata Dutta and Tanmoy Chakraborty},
      year={2024},
      eprint={2405.10548},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


