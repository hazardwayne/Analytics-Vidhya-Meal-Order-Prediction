## Solution of Genpact Machine Learning Hackathon at Analytics Vidhya platform

The target is to predict meal's order num in the next week, our solution ranks 30 over 765 participants, please learn more details at the [competition home page](https://datahack.analyticsvidhya.com/contest/genpact-machine-learning-hackathon/). 

## Methods

#### Feature Engineering

* Category Encoding: We simply transform them from *String* to *Category* then *Int* as they are all non-ordinal.

* Group Features: Based on *price*, *promotion* and our generated *category_hotness* *etc*, we aggragate over *meal*, *region* and their combination *etc*.

* Multi-dimensional Time Series Features: For each category group, we generate the *history* feature which records meal order statistics up to now.<br>
Meanwhile, the 2-nd order(difference) features are further produced.

#### Model

* We split the data based on the *week* timeline, where the first 135 weeks are used for train and the last 10 weeks for evaluation.

* We train *xgboost* and *lightgbm* models respectively, and sum their predictions with 0.65 and 0.35 weights as our final result.

## Contents

* Under src folder, the ipython notebook file is about our exploring procedure.

* **solution.py** is the final submission to the competition.

* To reproduce the result, it is necessary to create the **data** folder besides src and put all csv files into it.