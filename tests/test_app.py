import pytest

from wemoney_recommender.main import sort_for_user_date

def test_shape():
    df = sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d')
    # All posts included?
    assert df.shape[0] == 50


def test_shape_cutoff():
    df = sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d', '2022-01-25')
    # Does date cutoff work?
    assert df.shape[0] == 28


def test_categories():
    df1 = sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d')
    df2 = sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d', use_categories=True)
    # Does using categories still get al posts but in different order?
    assert df1.shape[0] == df2.shape[0]
    assert df1.index.isin(df2.index).all()
    assert (df1.index != df2.index).any()


def test_date_order():
    df = sort_for_user_date('84b367ec-3f12-4040-a0ee-ed37d924f80d')
    assert df.post_time.iat[0] > df.post_time.iat[-1]
