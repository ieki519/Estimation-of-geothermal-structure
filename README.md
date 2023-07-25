# grad_maxとminをround(-2)へ変更
# tishitsu_rankに変更
# grad_max_h_zとgrad_min_h_zを追加

# output_japan_last
# curie <580にもhyperparameterを付けた
# 800以下の制限なし
# hyper param curie (1/10000)から(1/1000)


# tishitsu rankについて
# all
           t
group_ja     
火成岩       1.0
堆積岩       2.0
その他       3.0
付加体       4.0
変成岩       5.0

# inter 0
            t
group_ja     
火成岩       1.0
堆積岩       2.0
その他       3.0
変成岩       4.0
付加体       5.0
            t
group_ja     
火成岩       1.0
堆積岩       2.0
その他       3.0
付加体       4.0
変成岩       5.0
            t
group_ja     
火成岩       1.0
堆積岩       2.0
その他       3.0
付加体       4.0
変成岩       5.0
            t
group_ja     
火成岩       1.0
堆積岩       2.0
その他       3.0
付加体       4.0
変成岩       5.0
            t
group_ja     
火成岩       1.0
堆積岩       2.0
その他       3.0
付加体       4.0
変成岩       5.0

# extra 0
            t
group_ja     
火成岩       1.0
堆積岩       2.0
付加体       3.0
その他       4.0
変成岩       5.0

```python
if "tishitsu_rank" in name:
    tishitsu_rank_dict = train_data.groupby("group_ja").mean()[["t"]].rank(ascending=False).to_dict()["t"]
    train_data["group_rank"]=train_data["group_ja"].replace(tishitsu_rank_dict)
    test_data["group_rank"]=test_data["group_ja"].replace(tishitsu_rank_dict)
    est_data["group_rank"]=est_data["group_ja"].replace(tishitsu_rank_dict)
    curie_data["group_rank"]=curie_data["group_ja"].replace(tishitsu_rank_dict)
    curie_data_580_ika["group_rank"]=curie_data_580_ika["group_ja"].replace(tishitsu_rank_dict)
    curie_data_580_izyou["group_rank"]=curie_data_580_izyou["group_ja"].replace(tishitsu_rank_dict)
```

# 0,1やっている

# PIやりなおし

extra lastでdepth800の考慮なし
grad_max がrandom forest出力
grad でもdepth800の考慮なし
PIは80でとりあえず作成