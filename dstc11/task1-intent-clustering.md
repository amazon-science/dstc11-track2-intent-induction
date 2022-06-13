# Task 1 - Intent Clustering

## Task
Participants will develop a system capable of clustering intents in human-to-human (H2H)
conversations between customer service agents and their customers.

During the testing phase, participants will be given a set of conversations labeled with agent and customer speaker 
roles as well as a list of turns where customers express intents to the agent.
The number of intents will not be provided in the testing phase, however each dataset will have between 5 and 50 
main intents.

They are expected to provide a cluster label for each input turn in JSONL format:
```json lines
{"turn_id": "turn_001_001", "predicted_label": "4"}
{"turn_id": "turn_001_008", "predicted_label": "4"}
{"turn_id": "turn_001_021", "predicted_label": "17"}
...
{"turn_id": "turn_903_001", "predicted_label": "9"}
```

They are allowed to use any publicly available data as well as the datasets provided as development
data (which includes ground truth cluster labels) for this track to develop their system.

## Evaluation
Each submission will be evaluated using metrics:
* **Accuracy (ACC)** - Accuracy computed after finding a 1:1 assignment of predicted to gold clusters.
* **Clustering Precision / Recall / F1** – assign each predicted cluster to the most frequent corresponding gold cluster
  (Precision) or each gold cluster to most frequent predicted cluster (Recall).
  See [Haponchyk et al. (2018)](https://aclanthology.org/D18-1254/) for more details.
* Auxiliary metrics including normalized mutual information (NMI) and the adjusted Rand index (ARI).
  See `evaluate.py` for more details.

Accuracy (ACC) will be the primary metric used for ranking system submissions.

## Data
In addition to development data provided as part of this task, participants are allowed to use any publicly available 
data for developing their system.
However, participants should be aware that test conversations will be similar in length and style to the development 
data (but in different domains).

## Baselines
We include two baseline systems that cluster embeddings of input turns with k-means.

To select a value for *k* (the number of clusters), clustering is performed for different values of *k* and the
clustering that results in the highest average [Silhouette score](https://en.wikipedia.org/wiki/Silhouette_(clustering))
is selected.
To speed up the search for optimal *k*, [hyperopt](https://github.com/hyperopt/hyperopt) is used.

The two baselines use averaged [GloVe](https://github.com/stanfordnlp/GloVe) embeddings and 
[all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) (a pre-trained sentence embedding
 model from [SentenceTransformers](https://github.com/UKPLab/sentence-transformers)) respectively.

## Rules
* Participation is welcome from any team (academic, corporate, non-profit, or government).
* Teams can participate in either or both sub-tracks.
* The identity of participants will **not** be published or made public by organizers.
In written results, teams will be identified as team IDs (e.g. team1, team2, etc). 
The organizers will verbally indicate the identities of all teams at the workshop chosen for communicating results.
* Participants may identify their own team in publications or presentations but may not reveal the identities of 
other teams.
* Participants are allowed to use any publicly available datasets, resources or models.
* Participants are **not** allowed to perform any manual examination or modification of the test data.
* All submitted outputs will be released to the public after the testing phase ends.