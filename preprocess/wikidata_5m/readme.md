### Text - Knowledge Triplet Alignment


**Sentence extraction and triplet align**

```bash
>>> python preprocess/wikidata_5m/align_triplet_text.py \
      --kg ../research.data/wikidata5m/wikidata5m_transductive_train.txt \
      --entity_vocab ../research.data/wikidata5m/wikidata5m_entity.txt \
      --relation_vocab ../research.data/wikidata5m/wikidata5m_relation.txt \
      --corpus ../research.data/wikidata5m/wikidata5m_text.txt \
      --num_workers 16 --output_dir ../research.data/wikidata5m/triplet_text_align_v1.0
  
>>> Entity amount: 4813491
>>> Relation amount: 825
>>> Generated 5683300 text-triplet pairs with 14930979 samples overlooked.
>>> Total 6461397 aligned sentences with 1.1369093660373375 per triplet.
 
```

**Ranking through sentence-transformer**

Currently unnecessary? Since two sentence for each triplet at most (may not properly aligned).

**Construct logical circle**

```bash
>>> python preprocess/wikidata_5m/construct_logical_circle.py \
      --kg wikidata5m/wikidata5m_transductive_train.txt \
      --id2ent wikidata5m/triplet_text_align_v1.0/id2ent.json \
      --max_depth 4 --num_workers 24 --output_dir wikidata5m/triplet_text_align_v1.0    
```

