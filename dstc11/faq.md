# DSTC11 Track 2 - Frequently Asked Questions

Q: I finished my registration, but not been contacted by the track organizers. How do I participate in the track?  
A: No confirmation is required. All data, baseline and evaluation scripts are publicly available in this repository.

---

Q: How do the tasks (Task 1 – Intent Clustering vs. Task 2 – Open Intent Induction) differ?  
A: The two tasks differ in the following ways:
1. Inputs – Task 1 pre-specifies clustering inputs, while Task 2 requires identifying relevant turns,
or using the provided `InformIntent` dialog act predictions as inputs.
2. Outputs – Task 1 requires assigning a label to every input, while Task 2 only requires producing training data
for an intent classifier.
3. Evaluation – Task 1 uses clustering evaluation, whereas Task 2 evaluation is performed
by training an intent classifier on the induced training data and evaluating the predictions on
an external test set.

---

Q: Will the number of gold intents be provided for either task?  
A: The number of gold intents will not be provided.
However, each dataset will have between 5 and 50 main intents.

---

Q: Do I have to participate in both tasks or can I participate in either of them?  
A: You can participate in either or both sub-tasks.
Participation is encouraged for both, especially since submissions for Task 1 can be applied
directly to Task 2 using the provided `InformIntent` predictions as inputs.
