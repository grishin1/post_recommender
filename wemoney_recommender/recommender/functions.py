import pandas as pd
import numpy as np
import re
from itertools import chain
from sklearn.metrics import jaccard_score
from typing import List


INTERESTS_TO_CATEGORIES_MAP = {
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

INTERESTS_TO_CATEGORIES_MAP = {k.lower(): v for k, v in INTERESTS_TO_CATEGORIES_MAP.items()}


TAGS_TO_INTERESTS_MAP = {
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


def extract_post_hashtags(tags: str) -> List[str]:
    res1 =  re.subn(r".*[[]|[]].*|[']|\s+", '', tags)[0]
    if res1.find(',') < 0:
        return res1.lower().split(';')
    else:
        return res1.lower().split(',')


def get_n_replies(_id: str, _df: pd.DataFrame) -> int:
    return _df.parent_id.eq(_id).sum()


def trim_unicodes(_str: str) -> str:
    return ''.join([l for l in list(_str) if l.isascii()]).strip()


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


def get_last_reply_recency_rank(post_id: str, par: pd.Series, _dfp: pd.DataFrame) -> int:
    if get_n_replies(post_id, _dfp) > 0:
        last_reply_id = _dfp[(_dfp.parent_id == post_id)].sort_values('post_time')['post_id'].iat[-1]
    else:
        last_reply_id = post_id
    return par[last_reply_id]


def get_non_pesonalised_rating_matrix(par: pd.Series, rrr: pd.Series, _dfp: pd.DataFrame) -> pd.DataFrame:
    # rrr -- raw replies ranks
    par = par.copy()
    par.name = 'recency_rank'
    raw = rrr.copy()
    raw.name = 'n_replies_rank'
    rep = par.index.to_series().apply(get_last_reply_recency_rank, args=(par, _dfp))
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


def get_rating_matrix(uid, um, ucm, pcm, _dfp, par, rrr, *, use_categories=False):
    mx1 = get_non_pesonalised_rating_matrix(par, rrr, _dfp)
    mx2 = get_personalised_rating_matrix(uid, um, ucm, pcm, _dfp, use_categories=use_categories)
    out = pd.merge(mx1, mx2, how='left', left_index=True, right_index=True)
    out['total_rank'] = out.apply(np.mean, axis=1)
    out.sort_values('total_rank', inplace=True)
    return out


def sort_posts_for_user(uid: str, um, ucm, pcm, par, rrr, _dfp, *, use_categories=False) -> pd.DataFrame:
    mx = get_rating_matrix(uid, um, ucm, pcm, _dfp, par, rrr, use_categories=use_categories)
    mx.sort_values('total_rank', inplace=True) #just in case
    sorted_ids = mx.index
    out = _dfp.copy().set_index('post_id').loc[sorted_ids]
    return out
