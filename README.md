# Intent Induction from Conversations for Task-Oriented Dialogue
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repository contains data, relevant scripts and baseline code for the [DSTC11](https://dstc11.dstc.community/)
summer track on *Intent Induction from Conversations for Task-Oriented Dialogue*.

This track aims to evaluate methods for the automatic induction of customer intents in the realistic setting of customer
service interactions between human agents and customers. As complete conversations will be provided, participants can
make use of information in both agent and customer turns. The track includes two tasks: (1) intent clustering, which
requires participants to assign labels to turns in the dialogues where customers express intents, and (2) open intent
induction, in which participants must induce a set of intents from dialogues, with each intent defined by a list of 
sample utterances to be used as training data for an intent classifier.

**Organizers:** James Gung, Raphael Shu, Jason Krone, Salvatore Romeo, Arshit Gupta, Yassine Benajiba, Saab Mansour and
Yi Zhang

**Contact:** dstc11-intent-induction (AT) amazon (DOT) com

## Timeline

* Development data release: June 13, 2022
* Test data release: Middle of September
* Submission of final results: End of September
* Final result announcement: Early of October
* Paper submission: Middle of November
* Workshop: February or March 2023

## DSTC11 Track 2 Tasks
* See [Task 1 - Intent Clustering](/dstc11/task1-intent-clustering.md) for more details on Task 1.
* See [Task 2 - Open Intent Induction](/dstc11/task2-open-intent-induction.md) for more details on Task 2.

## Running Baselines

Python 3 (`>=3.7`) is required. Using a conda/virtual environment is recommended.

```bash
# install dependencies
pip3 install -r requirements.txt

# run intent clustering (Task 1) baselines and evaluation
python3 -m sitod.run_experiment \
--data_root_dir dstc11 \
--experiment_root_dir results \
--config configs/run-intent-clustering-baselines.jsonnet

# run open intent induction (Task 2) baselines and evaluation
python3 -m sitod.run_experiment \
--data_root_dir dstc11 \
--experiment_root_dir results \
--config configs/run-open-intent-induction-baselines.jsonnet
```

## Important Links

* [Track Proposal](https://drive.google.com/file/d/1itlby2Ypq3sRVtOY1alr3ygjPZZdB2TT/view)
* [Challenge Registration](https://forms.gle/e2qVGPPAhpp8Upt8A)
* [DSTC11 Website](https://dstc11.dstc.community/)
* [DSTC Mailing List](https://groups.google.com/a/dstc.community/forum/#!forum/list/join)

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

