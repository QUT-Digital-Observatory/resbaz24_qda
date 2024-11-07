# Queensland Election Reddit Dataset

This dataset was collected from Reddit on the 30th of October 2024. It contains data posted on Reddit between the 23rd to the 30th of October. The Queensland state election took place on the 26 of October, and the Liberal National Party was declared to have won the election on the 27th of October.

This dataset is a subset of the total volume of submissions and comments posted in this period, selected based on the topic relevance to the Queensland state election. This was determined by a generative AI (see prompt below).

The dataset contains 25,787 comments drawn from 191 submissions. The following table shows how these were distributed across the 8 submissions in this collection:

| Subreddit      | Submissions | Comments |
|----------------|-------------|----------|
| queensland     | 104         | 13191    |
| brisbane       | 66          | 11747    |
| GoldCoast      | 6           | 306      |
| Townsville     | 3           | 156      |
| ipswich        | 5           | 166      |
| Toowoomba      | 2           | 121      |
| Cairns         | 2           | 51       |
| sunshinecoast  | 3           | 49       |

## Method

The subreddit list is determined by the 'qld' tags in subreddits in the Digital Observatory's AusReddit platform. The data itself was collected directly from the live Reddit API.

A list of 392 submissions submissions was filterd down to 191 submissions based on coding by a generative AI (see below).

We fetched all comments for the 191 submissions, resulting in 25,787 comments.

We counted the raw data and compared validated the filtering process by checking accumulated num_comment field in submissions with the total number of filtered comments. This differs by a few hundred comments which is normal due to the nature of Reddit's API (deleted and removed comments appear in the total counts).

## Data files

The data is provided in three [parquet](https://parquet.apache.org/) files:

- `subreddits.parquet`
- `submissions.parquet`
- `comments.parquet`

For Python, you can load these into dataframes using Pandas, for example:

```python
import pandas as pd
df = pd.read_parquet('comments.parquet')
```

For R, you will need the arrow package:

```R
install.packages("arrow")
library(arrow)
read_parquet("submissions.parquet")
```

If you want to open the files in a spreadsheet for some reason, you may want CSV files. This can be easily accomplished in Python with Pandas:

```python
import pandas as pd
pd.read_parquet('submissions.parquet').to_csv('submissions.csv', index=False)
```

## Data structure

The `subreddits.parquet` file contains the following fields:

| Field        | Description                                                                 |
|--------------|-----------------------------------------------------------------------------|
| id           | The unique identifier for the subreddit.                                    |
| display_name | The display name of the subreddit.                                          |
| subscribers  | The number of subscribers to the subreddit.                                 |
| description  | A brief description of the subreddit.                                       |

The `submissions.parquet` file contains the following fields:

| Field         | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| author        | The username of the Reddit user who created the submission.                 |
| id            | The unique identifier for the submission.                                   |
| title         | The title of the submission.                                                |
| permalink     | The URL path to the submission on Reddit.                                   |
| selftext      | The text content of the submission, if any.                                 |
| created_utc   | The timestamp when the submission was created, in UTC.                      |
| subreddit     | The name of the subreddit where the submission was posted.                  |
| num_comments  | The number of comments on the submission.                                   |


The `comments.parquet` file contains the following fields:

| Field         | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| id            | The unique identifier for the comment.                                      |
| author        | The username of the Reddit user who created the comment.                    |
| body          | The text content of the comment.                                            |
| created_utc   | The timestamp when the comment was created, in UTC.                         |
| root_comment  | A boolean indicating if the comment is a parent comment (not a reply).      |
| parent_id     | If root_comment True, the id of the submission otherwise a parent comment.  |
| subreddit     | The name of the subreddit where the comment was posted.                     |
| submission_id | The unique identifier of the submission to which the comment belongs.       |


## Generative AI 

We used Google's Gemini Flash 1.5 002 model to determine topic relevance with the state election. The coding prompt is as follows:

```
The following Reddit submissions are drawn from Australian subreddits in the state of Queensland. They were created in the last week, during which there was a Queensland state election. Please code these submissions as being relevant to the election (even generally) or not relevant. Output JSON as a simple object where the keys are the number of the item, and the value is true (for relevant) or false (for not relevant). Submissions follow:
```
