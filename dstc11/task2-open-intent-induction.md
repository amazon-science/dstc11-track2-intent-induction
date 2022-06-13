# Task 2 - Open Intent Induction

## Task
Participants will develop a system used to induce intent schemas for task-oriented dialogue from human-to-human (H2H)
conversations between customer service agents and their customers.

During the testing phase, participants will be given a set of conversations labeled with agent and customer speaker 
roles.
Each conversation will also include automatic `InformIntent` dialogue act predictions indicating the turns
where customers may have expressed an intent.
The number of intents will not be provided in the testing phase, however each dataset will have between 5 and 50 
main intents.

They are expected to provide an intent schema for each set of conversations in JSONL format:
```json lines
{"intent_id":  "intent_1", "utterances": ["I want to book a flight", "I need a flight", ...]}
{"intent_id":  "intent_2", "utterances": ["I want to reserve a room", "I need a hotel room", ...]}
...
```

They are allowed to use any publicly available data as well as the datasets provided as development
data for this track to develop their system.

**Notice:** During the testing phase, input turns for Task 1 (where customers provide intents) will not be available 
for use in Task 2.
Participants will be able to use the provided automatic dialog act classifier predictions as inputs to their system.

## Evaluation
Each submission will be evaluated using following methodology:

1. An intent classifier will be trained using the induced intent schema
   1. The intent classifier will use a pre-trained sentence encoder
   2. A softmax classifier will be trained on fixed embeddings from the sentence encoder
   3. For more details, see `evaluate.py`
2. Using the classifier, predictions will be made on a test set of utterances corresponding to ground truth intents 
present in the conversations
   1. `"Book me a flight" -> "intent_1"`
   2. `"Get a hotel room" -> "intent_2"`
3. Clustering metrics will be computed using the predicted induced intent labels and ground truth intent labels
   1. **Accuracy (ACC)** - Accuracy computed after finding a 1:1 assignment of predicted to gold clusters.
   2. **Clustering Precision / Recall / F1** – assign each predicted cluster to the most frequent corresponding gold cluster
  (Precision) or each gold cluster to most frequent predicted cluster (Recall).
  See [Haponchyk et al. (2018)](https://aclanthology.org/D18-1254/) for more details. 
   3. Auxiliary metrics including normalized mutual information (NMI) and the adjusted Rand index (ARI).
  See `evaluate.py` for more details.

Accuracy (ACC) will be the primary metric used for ranking system submissions.


## Data
In addition to development data provided as part of this task, participants are allowed to use any publicly available 
data or pre-trained models for developing their system.
However, participants should be aware that test conversations will be similar in length and style to the development 
data (but in different domains).

## Baselines
Task 2 re-uses the same clustering baselines from [Task 1](/dstc11/task1-intent-clustering.md).

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