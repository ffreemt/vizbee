"""Test align_sents."""
from pyvizbee.align_sents import align_sents
from seg_text import seg_text

text1 = """`Wretched inmates!' I ejaculated mentally, `you deserve perpetual isolation from your species for your churlish inhospitality. At least, I would not keep my doors barred in the day time. I don't care--I will get in!' So resolved, I grasped the latch and shook it vehemently. Vinegar-faced Joseph projected his head from a round window of the barn."""
text2 = """“被囚禁的囚犯!”我在精神上被射精,“你应该永远与你的物种隔绝,因为你这种粗鲁的病态。至少,我白天不会锁门,我不在乎,我进去了!”我决心如此,我抓住了门锁,狠狠地摇了一下。醋脸的约瑟夫从谷仓的圆窗朝他的头照射。"""
text3 = """"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit. Zumindest würde ich meine Türen tagsüber nicht verriegeln. Das ist mir egal - ich werde reinkommen!' So entschlossen, ergriff ich die Klinke und rüttelte heftig daran. Der essiggesichtige Joseph streckte seinen Kopf aus einem runden Fenster der Scheune."""


def test_align_sents_sanity():
    """Test align_sents sanity check."""
    lst1, lst2 = [
        "a",
        "bs",
    ], ["aaa", "34", "a", "b"]
    res = align_sents(lst1, lst2)

    assert res == [("a", "aaa"), ("a", "34"), ("bs", "a"), ("bs", "b")]


def test_align_sents_en_zh():
    """Test align_sents en-zh."""
    sents_en = seg_text(text1)
    sents_zh = seg_text(text2)

    # 9ms vs shuffle_sents 50ms shuffle_sents wth lang1lang2 40ms
    res = align_sents(sents_en, sents_zh)

    _ = """res[2:4]
    Out[26]:
    [('At least, I would not keep my doors barred in the day time.',
      '至少,我白天不会锁门,我不在乎,我进去了!”'),
     ("I don't care--I will get in!'", '至少,我白天不会锁门,我不在乎,我进去了!”')]
    """
    assert "至少" in str(res[2])
    assert "至少" in str(res[3])


def test_align_sents_en_de():
    """Test align_sents en-zh."""
    sents_en = seg_text(text1)
    sents_de = seg_text(text3)

    res1 = align_sents(sents_en, sents_de)
    _ = """In [48]: res1[:2]
    Out[48]:
    [("`Wretched inmates!'",
      '"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit.'),
     ('I ejaculated mentally, `you deserve perpetual isolation from your species for your churlish inhospitality.',
      '"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit.')]
    """
    assert "Elende" in str(res1[0])
    assert "Elende" in str(res1[1])


_ = """
[("`Wretched inmates!'",
  '"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit.'),
 ('I ejaculated mentally, `you deserve perpetual isolation from your species for your churlish inhospitality.',
  '"Elende Insassen! ejakulierte ich im Geiste, "ihr verdient die ewige Isolation von eurer Spezies für eure rüpelhafte Ungastlichkeit.'),
 ('At least, I would not keep my doors barred in the day time.',
  'Zumindest würde ich meine Türen tagsüber nicht verriegeln.'),
 ("I don't care--I will get in!'",
  "Das ist mir egal - ich werde reinkommen!'"),
 ('So resolved, I grasped the latch and shook it vehemently.',
  'So entschlossen, ergriff ich die Klinke und rüttelte heftig daran.'),
 ('Vinegar-faced Joseph projected his head from a round window of the barn.',
  'Der essiggesichtige Joseph streckte seinen Kopf aus einem runden Fenster der Scheune.')]


"""
