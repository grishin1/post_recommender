import pandas as pd
import numpy as np
import re
from itertools import chain
from sklearn.metrics import jaccard_score
from typing import List


dfu = pd.read_csv("./data/users.csv", parse_dates=['dob'], infer_datetime_format=True)
dfi = pd.read_csv("./data/interest.csv")
dfp = pd.read_csv("./data/posts.csv", parse_dates=['post_time'], infer_datetime_format=True)
dfp.hashtags = dfp.hashtags.str.lower()


def extract_post_hashtags(tags: str) -> List[str]:
    res1 =  re.subn(r".*[[]|[]].*|[']|\s+", '', tags)[0]
    if res1.find(',') < 0:
        return res1.lower().split(';')
    else:
        return res1.lower().split(',')


all_tags = []
for t in dfp.hashtags.apply(extract_post_hashtags):
    all_tags.extend(t)


all_tags = set(all_tags)


all_n_replies = dfp.groupby('post_id')['post_id'].agg(lambda x: (x.item() == dfp.parent_id).sum())
all_n_replies.name = 'n_replies'

def get_n_replies(_id: str, _df: pd.DataFrame) -> int:
    return _df.parent_id.eq(_id).sum()


post_age = (pd.Timestamp.now() - dfp.set_index('post_id').post_time).dt.days
post_age.name = 'post_age'


def trim_unicodes(_str: str) -> str:
    return ''.join([l for l in list(_str) if l.isascii()]).strip()


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


interests_to_categories_map = {
    'Reducing spending': 'saving',
    'See my spending habits': 'saving',
    'General budgeting': 'saving',
    'Budgeting with kids': 'saving',

    'Track bills': 'budgeting',
    'Save for a car': 'budgeting',
    'Buy a property': 'budgeting',
    'Deals and discounts': 'budgeting',
    'Bargains & Sales': 'budgeting',

    'Side Hustles': 'lifestyle',
    'FI/RE': 'lifestyle',
    'Debt free living': 'lifestyle',

    'Reduce my rates on my personal loan': 'loan',
    'Credit score improvement': 'loan',
    'Paying Down Debt': 'loan',

    'Groceries': 'other',
    'Money facts and trivia': 'other',
    'Getting married': 'other',

    'Investing': 'investing',
    'Stocks': 'investing',
    'Crypto': 'investing',
    'ETFS': 'investing'
}

interests_to_categories_map = {k.lower(): v for k, v in interests_to_categories_map.items()}


tags_to_interests_map = {
    '24yearsold': 'lifestyle',
    'financialfreedom': 'lifestyle',
    'financialindependance': 'lifestyle',
    'networth': 'lifestyle',
    'moretime': 'lifestyle',
    'mumsmakingmoney': 'lifestyle',
    'mumsthathustle': 'lifestyle',
    'sidehustle': 'lifestyle',
    'sidehustlingmums': 'lifestyle',
    'workhard': 'lifestyle',

    'saving': 'saving',
    'savings': 'saving',
    'wealth': 'saving',

    'budget': 'budgeting',
    'spending': 'budgeting',
    'expenses': 'budgeting',
    'tax': 'budgeting',

    'loan': 'loan',
    'debt': 'loan',
    'credit': 'loan',

    'plantingourmoneytree': 'other',
    'rookie': 'other',
    '': 'other',
    'other': 'other',

    'investment': 'investing',
    'finance': 'investing'
}


categories_list = dfi.groupby('interest').agg(list)
categories_list.index = categories_list.index.to_series().apply(lambda x: trim_unicodes(x).lower())
categories_list['category'] = categories_list.index.map(interests_to_categories_map)
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
posts_cat_list['category'] = posts_cat_list.hashtags.apply(lambda x: list(set([tags_to_interests_map[el] for el in extract_post_hashtags(x)])))
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


def one_vs_rest_jaccard(_id: str, col_name: str, _df: pd.DataFrame) -> pd.Series:
    df = _df.copy().set_index(col_name)
    v1 = df.loc[_id].to_list()
    res = df.drop(_id)
    for idx in res.index:
        res.loc[idx, 'jaccard'] = jaccard_score(v1, df.loc[idx].to_list())
    return res.sort_values('jaccard', ascending=False).loc[:, 'jaccard']


def user_vs_post_jaccard(uid: str, _udf: pd.DataFrame, _pdf: pd.DataFrame) -> pd.Series:
    v1 = _udf.set_index('uid').loc[uid].to_list()
    res = _pdf.set_index('post_id').copy()
    for idx, vec in res.iterrows():
        res.loc[idx, 'jaccard'] = jaccard_score(v1, vec.to_list())
    return res.sort_values('jaccard', ascending=False)['jaccard']


post_age_ranks = post_age.sort_values().rank(method='min')

raw_replies_ranks = all_n_replies.sort_values(ascending=False).rank(method='min', ascending=False)

def posts_ranks_by_user_interests(uid: str, ucm: pd.DataFrame, pcm: pd.DataFrame) -> pd.Series:
    similarity = user_vs_post_jaccard(uid, ucm, pcm)
    return similarity.rank(method='dense', ascending=False)


def posts_ranks_from_similar_users(uid: str, um: pd.DataFrame, ucm: pd.DataFrame, _dfp: pd.DataFrame, *, use_categories=False) -> pd.Series:
    if use_categories:
        similar_users = one_vs_rest_jaccard(uid, 'uid', ucm)
    else:
        similar_users = one_vs_rest_jaccard(uid, 'uid', um)
    ranked_posts = pd.merge(similar_users, _dfp.loc[:, ['uid', 'post_id']], how='left', left_index=True, right_on='uid')
    ranks = ranked_posts.set_index('post_id').jaccard.rank(method='dense', ascending=False)
    return ranks


def get_last_reply_recency_rank(post_id: str, _dfp: pd.DataFrame) -> int:
    if get_n_replies(post_id, _dfp) > 0:
        last_reply_id = _dfp[(_dfp.parent_id == post_id)].sort_values('post_time')['post_id'].iat[-1]
    else:
        last_reply_id = post_id
    return post_age_ranks[last_reply_id]


def get_non_pesonalised_rating_matrix() -> pd.DataFrame:
    par = post_age_ranks.copy()
    par.name = 'recency_rank'
    raw = raw_replies_ranks
    raw.name = 'n_replies_rank'
    rep = par.index.to_series().apply(get_last_reply_rank, args=(dfp,))
    out = pd.merge(par, raw, how='left', left_index=True, right_index=True)
    out = pd.merge(out, rep, how='left', left_index=True, right_index=True)
    out.rename(columns={'post_id': 'reply_recency_rank'}, inplace=True)
    return out


def get_personalised_rating_matrix(uid: str, um: pd.DataFrame, ucm: pd.DataFrame, pcm: pd.DataFrame, _dfp: pd.DataFrame, *, use_categories=False) -> pd.DataFrame:
    ranks_by_user_interest = posts_ranks_by_user_interests(uid, ucm, pcm)
    ranks_by_user_interest.name = 'user_interest_rank'
    ranks_by_similar_user = posts_ranks_from_similar_users(uid, um, ucm, _dfp, use_categories=use_categories)
    ranks_by_similar_user.name = 'similar_user_rank'
    out = pd.merge(ranks_by_user_interest, ranks_by_similar_user, how='left', left_index=True, right_index=True)
    return out


def get_rating_matrix(uid, um, ucm, pcm, _dfp, *, use_categories=False):
    mx1 = get_non_pesonalised_rating_matrix()
    mx2 = get_personalised_rating_matrix(uid, um, ucm, pcm, _dfp, use_categories=use_categories)
    out = pd.merge(mx1, mx2, how='left', left_index=True, right_index=True)
    out['total_rank'] = out.apply(np.mean, axis=1)
    out.sort_values('total_rank', inplace=True)
    return out


def sort_posts_for_user(uid: str, um, ucm, pcm, _dfp, *, use_categories=False) -> pd.DataFrame:
    mx = get_rating_matrix(uid, um, ucm, pcm, _dfp, use_categories=use_categories)
    mx.sort_values('total_rank', inplace=True) #just in case
    sorted_ids = mx.index
    out = _dfp.copy().set_index('post_id').loc[sorted_ids]
    return out
