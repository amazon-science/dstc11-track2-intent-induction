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

## News
* **September 26th** - Test conversations released!
  * Two domains: [Banking](/dstc11/test-banking) and [Finance](/dstc11/test-finance)
  * [Submission form](https://forms.gle/m2NWYm22LGGyEYtn9) open until **October 3rd, 2022 (11:59pm Anywhere on Earth UTC-12)** 
  * Submission [standalone validation script](/dstc11/task1-intent-clustering.md#Submissions) provided (see [Submissions](/dstc11/task1-intent-clustering.md#Submissions))
* **October 24th** - Ground truth intent labels, test utterances, all submitted entries, and raw results released
  * [Task 1 Submissions](/dstc11/dstc11-submissions/1) and [Results](https://docs.google.com/spreadsheets/d/1QV3ZyodLkttaGAXFDVAKYKIFZW7X3hzdysaDhWOrpqA/edit?usp=sharing)
  * [Task 2 Submissions](/dstc11/dstc11-submissions/2) and [Results](https://docs.google.com/spreadsheets/d/15K3vBDfAj_fzqK988rrESRuk4MTgyIOKIarKGZpGja4/edit?usp=sharing)
## Timeline

* Development data release: June 13, 2022
* Test data release: September 26th, 2022
* Entry submission deadline: October 3rd, 2022
* Final result announcement: October 24th, 2022
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

