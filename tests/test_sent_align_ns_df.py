"""Test sent_align_ns_df."""
# pylint: disable=invalid-name

import tests.ns_df as ns_df
from pyvizbee.sent_align_ns_df import sent_align_ns_df

df_ = ns_df.df_.copy()
model = ns_df.model

assert len(df_) > 30


def test_align_ns_df():
    """Test sent_align_ns_df."""
    # lst = df_.metric[df_.metric.replace("", float("-inf")) < 0].index.to_list()
    # sl = locate_index_pairs(lst)

    _ = sent_align_ns_df(df_)
    # assert _.metric.to_list()[28] > 0.4  # .48

    # In [424]: print(str(_.iloc[51]))
    # text1     I commenced again.
    # text2               我又开始说话。“
    # metric
    # Name: 51, dtype: object
    assert "commence" in _.iloc[51].text1
    assert "开始" in _.iloc[51].text2

    # In [427]: print(str(_.iloc[53]))
    # text1     `They are not mine,' said the amiable hostess,...
    # text2            “那些不是我的，”这可爱可亲的女主人说，比希刺克厉夫本人所能回答的腔调还要更冷淡些。
    # metric 0.28
    # Name: 53, dtype: object
    assert "amiable" in _.iloc[53].text1
    assert "可亲的" in _.iloc[53].text2
    assert _.iloc[53].metric > 0.2  # 0.28
