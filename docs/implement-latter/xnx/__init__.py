"""
XnX-enhanced PathRAG module.

This module extends the BUPT-GAMMA PathRAG implementation with XnX notation
for weighted path tuning in knowledge graphs.

Citation for original PathRAG:
@article{chen2025pathrag,
  title={PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths},
  author={Chen, Boyu and Guo, Zirui and Yang, Zidan and Chen, Yuluo and Chen, Junze and Liu, Zhenghao and Shi, Chuan and Yang, Cheng},
  journal={arXiv preprint arXiv:2502.14902},
  year={2025}
}
"""

from .xnx_pathrag import XnXPathRAG
from .xnx_params import XnXQueryParams, XnXIdentityToken
from .arango_adapter import ArangoPathRAGAdapter
