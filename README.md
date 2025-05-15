# Amazon Food Reviews Analysis

## *Introduction*
This projects an exploratory analysis of Amazon food products reviews using **Python** and **Power Bi**. The main goal here is to uncover insights about costumer sentiment, sales trend, and user behavior. All of this based on thousands of real-world product reviews (1999-2012).

The dataset covered here is from Kaggle and you can download it and analyse it yourself here: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

##  *Dataset Summary*
This dataset contains thousands of rows, each representing a customer review with variables such as score, review text, user ID, product ID, timestamp, helpfulness ratings, and more. Below is a list of all available columns:

| Column Name             | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| Id                      | Row Id                                                                      |
| ProductId               | Unique identifier for the product                                           |
| UserId                  | Unique identifier for the user                                              |
| ProfileName             | Profile name of the user                                                    |
| HelpfulnessNumerator    | Number of users who found the review helpful                                |
| HelpfulnessDenominator  | Number of users who indicated whether they found the review helpful or not  |
| Score                   | Rating between 1 and 5                                                      |
| Time                    | Timestamp for the review                                                    |
| Summary                 | Brief summary of the review                                                 |
| Text                    | Text of the review                                                          |

As you can see, the dataset contains valuable information that can lead us to several interesting questions, such as:

- What is the all-time average score?

- Which words in the review text are most associated with high or low scores?

- How has the frequency of reviews changed over the years?

- Which product IDs have the highest average ratings?

In this analysis, we’ll explore patterns and insights based on key questions.

## *Data Preparation & Tools*
As mentioned earlier, the two main tools used throughout this project are Python and Power BI—both essential in the toolkit of any data analyst. While Power BI is fully capable of handling data preparation tasks, I chose to use Python for the data cleaning and modeling stages, as I find myself more proficient and flexible with it for these purposes.

Another key tool in this process was JupyterLab, which allowed me to write and execute code in well-organized blocks, making the workflow clearer and more efficient. You can view all the .ipynb notebooks used in this analysis in the [scripts](./scripts/) folder.



### Preparation 1: Python

In the [scripts](./scripts/) folder, you can check all the processing script in the [data_processing.ipynb](./scripts/data_processing.ipynb) file.

To begin the data preparation process, i downloaded the .csv file from the kaggle link mentioned before and imported using "read_csv()" function.
```python
import pandas as pd
data = read_csv(path)
print(data)
```
Although I could have imported the dataset using other methods—such as the Kaggle API—I chose a simpler approach for this case by manually downloading the file. The data is now loaded and accessible through the data variable, ready for transformation and analysis.

One important preprocessing step was removing rows with NULL values in essential columns to ensure the integrity of each record. Additional cleaning operations included validating the Score column to remove values outside the 0–5 range, and checking for inconsistencies in the HelpfulnessNumerator and HelpfulnessDenominator fields.

```python
t_data = data.copy()
# Removing Null Rows
t_data = t_data.dropna(subset=['ProductId', 'UserId', 'Time', 'Text'])

# Removing Invalid Scores
t_data = t_data[(t_data['Score'] >= 0) & (t_data['Score'] <= 5)]

# Removing invalid helpfulness
t_data = t_data[t_data['HelpfulnessNumerator'] <= t_data['HelpfulnessDenominator']]
```

If you examine the columns in the dataset, you'll notice that it includes a HelpfulnessNumerator, which indicates the number of users who found a review helpful. However, it doesn't explicitly show how many users did not find the review helpful.

To address this and make the data easier to analyze, I created a new column that calculates the difference between HelpfulnessDenominator and HelpfulnessNumerator. This new column directly represents the number of users who did not find the review helpful, providing a clearer and more interpretable metric for our analysis:

```python
negativeHelpfulness = data['HelpfulnessDenominator'] - data['HelpfulnessNumerator']
pos = data.columns.get_loc('HelpfulnessNumerator')
t_data.insert(pos+1, 'NegativeHelpfulness', negativeHelpfulness)
t_data = t_data.rename(columns={'HelpfulnessNumerator': 'PositiveHelpfulness'})
t_data = t_data.rename(columns={'HelpfulnessDenominator': 'TotalHelpfulness'})
```

Now we have two clearer columns: PositiveHelpfulness and NegativeHelpfulness, which make the data much easier to interpret.

Regarding the time aspect, the Time column in the original dataset was not in a proper datetime format—it was stored as an object type. To resolve this, I converted the column to a valid datetime format using the appropriate transformation:

```python
t_data['Time'] = pd.to_datetime(t_data['Time'], unit='s', utc=True)
```
Now all the date values are available for search and operations.

Finally, the dataset can be saved in a .csv format:
```python
t_data.to_csv('dataset/Reviews_t.csv')
```

As you can see in the scripts folder, there are four additional notebooks. Each of them was created to transform the data into specific tables tailored for easier use and analysis in Power BI.

### Preparation 2: Power Bi

