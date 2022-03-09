### Installation

This post recommender is a standard python package which can also be used as a CLI app (not tested).

To start using the recommender you can:
1.  Clone this repo

2.  pip-install it from Github using `pip install git+https://github.com/grishin1/post_recommender.git@master`

3. To run the tests, clone the repo, `cd` into `wemoney_recommender` directory (containing setup.py) and run `python -m pytest`

4. Consider installing into a virtual environment

### How to use

1.  The simplest option would be using top level function `sort_for_user_date` from Jupyter
    -  In Jupyter notebook `cd` into `wemoney_recommender` directory (containing setup.py)
    -  run `from wemoney_recommender.main import sort_for_user_date`
    - read the doc string by running `sort_for_user_date?` command
    - Start using the function with valid uid, e.g. `sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d')`,   
    `sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d', '2022-01-25')`    
    `sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d', use_categories=True)`

2.  It should also be possible to use the code from CLI after pip-installing the package or running `python setup.py install` (doesn't work on Win, should work on MacOS, didn't test) which installs `sort_posts` CLI script.
    -  in the terminal run `sort_posts 84b367ec-3f12-4040-a0ee-ed37d924f80d`
    - run `sort_posts -h` to see more arguments

### Design

The design follows closely bullet points from the assignment.

1.  The nature of the content puts a lot of emphasis on non-personalised recommendations (recency, number of replies and recency of replies) as well as personalised part where user interests and content from similar users are accounted for. Thus the resulting recommender is a hybrid recommendation system combining these requirements.

2.  User interests are taken into account by mapping user interests to posts' hashtags. For posts with missing hastags, parsing the post text is used to match the text to existing tags. For this part similarity of user interest to posts is calculated.

3. Similar users are found by collaborative filtering-like method where posts authored by similar authors are recommened. Only user interests are used for similarity calculation. In principle `age` from users CSV could have been used as well (not implemented).

4.  Rankings from different parts were combined in the rating matrix using simple averaging.

5. No filtering of posts according to interests was implemented because of sparsity of the data.    `use_categories` argumend was implemented to recommend based on more general categories. Thus both user interests and hashtags were fused into broader categories (e.g. 'crypto', 'stocks', 'ETFS' into 'investing').

6.  Date cutoff argument allows 'time travel' by considering only posts which existed up to that date.
