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
    - Start using the function with valid uid, e.g. sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d')

2.  It should also be possible to use the code from CLI after pip-installing the package or running `python setup.py install` (doesn't work on Win, should work on MacOS, didn't test) which installs `sort_posts` CLI script.
    -  in the terminal run `sort_posts 84b367ec-3f12-4040-a0ee-ed37d924f80d`
    - run `sort_posts -h` to see more arguments

### Design
