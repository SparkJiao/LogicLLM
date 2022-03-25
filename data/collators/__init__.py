"""
Write your own your own collators under the directory.
"""
from .tuple2dict import Tuple2DictCollator, NLITuple2DictCollator
from .wiki import WikiPathDatasetCollator, WikiPathDatasetCollatorOnlyMLM, WikiPathDatasetCollatorWithContext
