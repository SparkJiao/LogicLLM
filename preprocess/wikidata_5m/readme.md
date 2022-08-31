### Text - Knowledge Triplet Alignment


**Sentence extraction and triplet align**

```bash
>>> python align_triplet_text.py \
    --kg ../../wikidata5m/wikidata5m_transductive_train.txt \
    --entity_vocab ../../wikidata5m/wikidata5m_entity.txt \
    --relation_vocab ../../wikidata5m/wikidata5m_relation.txt \
    --corpus ../../wikidata5m/wikidata5m_text.txt \
    --num_workers 32 --output_dir ../../wikidata5m/triplet_text_align_v1.0
>>> Entity amount: 4813491
>>> Relation amount: 825
>>> aligning sentences: 100%|██████████████| 20614279/20614279 [24:20<00:00, 14115.30it/s]
>>> Generated 4218775 text-triplet pairs with 16395504 samples overlooked.
>>> Total 4544898 aligned sentences with 1.0773027715391317 per triplet.


$ source run_triplet_align.sh
Generated 15323472 text-triplet pairs with 5290807 samples overlooked.
Total 21042188 aligned sentences with 1.3731997552512902 per triplet.
  
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
    --min_depth 3 --max_depth 4 --num_workers 16 \
    --output_dir wikidata5m/triplet_text_align_v1.0/logical_circle/ --split_chunk 25,0
>>> 4568492 20600156 20600156
>>> Searching path: 100%|██████████████████████| 182739/182739 [1:03:46<00:00, 47.76it/s]
>>> Generate 147498326 paths.
```

