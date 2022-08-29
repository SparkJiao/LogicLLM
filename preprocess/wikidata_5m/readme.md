### Text - Knowledge Triplet Alignment


**Sentence extraction and triplet align**

```bash
>>> python align_triplet_text.py \
      --kg ../../wikidata5m/wikidata5m_transductive_train.txt \
      --entity_vocab ../../wikidata5m/wikidata5m_entity.txt \
      --relation_vocab ../../wikidata5m/wikidata5m_relation.txt \
      --corpus ../../wikidata5m/wikidata5m_text.txt \
      --num_workers 16 --output_dir ../../wikidata5m/triplet_text_align_v1.0
>>> Entity amount: 4813491
>>> Relation amount: 825
>>> aligning sentences: 100%|████████████████████| 20614279/20614279 [38:20<00:00, 8962.38it/s]
>>> Generated 4778788 text-triplet pairs with 15835491 samples overlooked.
>>> Total 5272402 aligned sentences with 1.1032927177351244 per triplet.
  
#>>> Entity amount: 4813491
#>>> Relation amount: 825
#>>> Generated 5683300 text-triplet pairs with 14930979 samples overlooked.
#>>> Total 6461397 aligned sentences with 1.1369093660373375 per triplet.

```

**Ranking through sentence-transformer**

Currently unnecessary? Since two sentence for each triplet at most (may not properly aligned).

**Construct logical circle**

```bash
>>> python preprocess/wikidata_5m/construct_logical_circle.py \
      --kg wikidata5m/wikidata5m_transductive_train.txt \
      --id2ent wikidata5m/triplet_text_align_v1.0/id2ent.json \
      --max_depth 4 --num_workers 16 --output_dir wikidata5m/triplet_text_align_v1.0 \
      --split_chunk 5,0
>>> 4568492 20600156 20600156
>>> Searching path: 100%|███████████████████████████████| 913698/913698 [4:42:50<00:00, 53.84it/s]
>>> Generate 22994740 paths.
```

