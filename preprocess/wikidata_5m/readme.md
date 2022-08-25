### Text - Knowledge Triplet Alignment


**Sentence extraction and triplet align**

```bash
python preprocess/wikidata_5m/align_triplet_text.py \
  --kg ../research.data/wikidata5m/wikidata5m_transductive_train.txt \
  --entity_vocab ../research.data/wikidata5m/wikidata5m_entity.txt \
  --relation_vocab ../research.data/wikidata5m/wikidata5m_relation.txt \
  --corpus ../research.data/wikidata5m/wikidata5m_text.txt \
  --num_workers 16 --output_dir ../research.data/wikidata5m/triplet_text_align_v1.0
```

**Ranking through sentence-transformer**


