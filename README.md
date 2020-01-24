# MLScoringLanguageTest

In this project, we provide an automated system for simulate the grading of Spanish L2 speaking which was done manually by human raters.
We rely on Natural Language Processing techniques and Machine Learning techniques to implement our solution. Given an input as response from test taker in textual format,
our engine called MLScoring will convert this information to a vector represented for the textual and Part of Speech tag of the input. We apply
Machine Learning algorithms to learn the correlation between the input vector and the predicted score for test takers. The grading scale is divided to 3 levels: Novice, Intermediate and Advance,
while each levels has 4 types:  MF, MM, SE, NE. Besides, our approach is available for predicting the topic of the response from test takers.
We evaluate MLScoring by 2 directions: using 10 fold cross validation and using training data for 2018 dataset and testing data as 2019 dataset both provided by the ETI company.
We achieve promising results for both the topic classification and score classification in these directions for 3 levels: Novice, Advance and Intermediate.
The best Machine Learning algorithms we had from the evaluation are LDA and Gradient Boosting, which are over 91% in average precision.
