# -*- encoding: utf-8 -*-
"""
H2O Segment Models.

:copyright: (c) 2020 H2O.ai
:license:   Apache License Version 2.0 (see LICENSE for details)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from h2o.base import Keyed
from h2o.frame import H2OFrame
from h2o.expr import ExprNode
from h2o.expr import ASTId

__all__ = ("H2OSegmentModels", )


class H2OSegmentModels(Keyed):

    #-------------------------------------------------------------------------------------------------------------------
    # Construction
    #-------------------------------------------------------------------------------------------------------------------

    def __init__(self, segment_models_id=None):
        self._segment_models_id = segment_models_id

    @property
    def key(self):
        return self._segment_models_id

    def detach(self):
        self._segment_models_id = None

    def as_frame(self):
        return H2OFrame._expr(expr=ExprNode("segment_models_as_frame", ASTId(self._segment_models_id)))._frame(fill_cache=True)
