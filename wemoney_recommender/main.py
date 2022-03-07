import argparse
import pkg_resources

from pandas import DataFrame, read_csv
from wemoney_recommender.recommender.functions import *


def sort_for_user_date() -> DataFrame:
    # Loading data:
    pth = pkg_resources.resource_filename("wemoney_recommender", "data")
    dfu = read_csv(pth + "/users.csv", parse_dates=['dob'], infer_datetime_format=True)
    dfi = read_csv(pth + "/interest.csv")
    dfp = read_csv(pth + "/posts.csv", parse_dates=['post_time'], infer_datetime_format=True)
    dfp.hashtags = dfp.hashtags.str.lower()


    all_tags = []
    for t in dfp.hashtags.apply(extract_post_hashtags):
        all_tags.extend(t)
    all_tags = set(all_tags)

    all_n_replies = dfp.groupby('post_id')['post_id'].agg(lambda x: (x.item() == dfp.parent_id).sum())
    all_n_replies.name = 'n_replies'

    post_age = (pd.Timestamp.now() - dfp.set_index('post_id').post_time).dt.days
    post_age.name = 'post_age'

    interest_list = dfi.groupby('interest').agg(list)
    interest_list.index = interest_list.index.to_series().apply(lambda x: trim_unicodes(x).lower())

    cols = interest_list.index.tolist()
    user_matrix = pd.DataFrame(columns=['uid'] + cols)
    for ind, val in enumerate(dfi.uid.unique()):
        for interest in cols:
            user_matrix.loc[ind, 'uid'] = val
            if val in interest_list.loc[interest, 'uid']:
                user_matrix.loc[ind, interest] = 1
            else:
                user_matrix.loc[ind, interest] = 0

    # Hashtags enrichment. Finding words in post that match existing hashtags:
    empty_ids = dfp.index[dfp.hashtags == "[]"]
    tags_from_post_text = dict.fromkeys(empty_ids, None)
    tag_list = list(all_tags)
    tag_list.remove('')

    for tag in tag_list:
        res1 = dfp.text.str.lower().str.find(tag)
        for idx in res1.index[res1 > -1]:
            if idx in empty_ids:
                if tags_from_post_text[idx] is None:
                    tags_from_post_text[idx] = [tag]
                else:
                    tags_from_post_text[idx].extend([tag])

    for idx, val in tags_from_post_text.items():
        if val is None:
            dfp.loc[idx, 'hashtags'] = "['other']"
        else:
            dfp.loc[idx, 'hashtags'] = str(val)

    categories_list = dfi.groupby('interest').agg(list)
    categories_list.index = categories_list.index.to_series().apply(lambda x: trim_unicodes(x).lower())
    categories_list['category'] = categories_list.index.map(INTERESTS_TO_CATEGORIES_MAP)
    categories_list = categories_list.groupby('category').agg(lambda x: list(set(chain.from_iterable(x))))

    cat_cols = categories_list.index.unique().tolist()
    user_cat_matrix = pd.DataFrame(columns=['uid'] + cat_cols)

    for ind, val in enumerate(dfi.uid.unique()):
        for cat in cat_cols:
            user_cat_matrix.loc[ind, 'uid'] = val
            if val in categories_list.loc[cat, 'uid']:
                user_cat_matrix.loc[ind, cat] = 1
            else:
                user_cat_matrix.loc[ind, cat] = 0

    posts_cat_list = dfp.loc[:, ['post_id', 'hashtags']]
    posts_cat_list['category'] = posts_cat_list.hashtags.apply(lambda x: list(set([TAGS_TO_INTERESTS_MAP[el] for el in extract_post_hashtags(x)])))
    posts_cat_list = posts_cat_list.explode('category')
    posts_cat_list = posts_cat_list.groupby('category')['post_id'].agg(list).to_frame()

    # Will re-user cat_cols:
    posts_cat_matrix = pd.DataFrame(columns=['post_id'] + cat_cols)

    for ind, val in enumerate(dfp.post_id.unique()):
        for cat in cat_cols:
            posts_cat_matrix.loc[ind, 'post_id'] = val
            if val in posts_cat_list.loc[cat, 'post_id']:
                posts_cat_matrix.loc[ind, cat] = 1
            else:
                posts_cat_matrix.loc[ind, cat] = 0


    post_age_ranks = post_age.sort_values().rank(method='min')
    raw_replies_ranks = all_n_replies.sort_values(ascending=False).rank(method='min', ascending=False)

    out = sort_posts_for_user('7e4cb900-a09e-4d1d-8b65-ecd3ef51f132', user_matrix, user_cat_matrix, posts_cat_matrix, post_age_ranks, raw_replies_ranks, dfp)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description='Sort user posts')
    parser.add_argument('uid', type=str, nargs=1, help='user_id as string')
    parser.add_argument('-c', '--categories', action='store_true', help='If used user interests are combined in broader categories')
    args = parser.parse_args()
    print(args.uid)
    print(args.categories)
    return None


if __name__ == '__main__':
    main()