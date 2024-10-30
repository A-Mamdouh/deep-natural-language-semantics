from src.logic.base.calculus import generate_models
from src.logic.base.syntax import Exists, Forall, Formula, Predicate, Sort, Variable
from src.logic.base.tableau import Tableau
from src.query_environment.environment import AxiomsBase, AxiomUtils

from dataclasses import dataclass, field
import functools
import json
import operator as op
from typing import Any, Dict, List, Optional, Tuple


class Sorts:
    individual: Sort = Sort("individual")
    verb: Sort = Sort("verb")
    event: Sort = Sort("event")
    adjective: Sort = Sort("adjective")
    adverb: Sort = Sort("adverb")


class Predicates:
    subject = Predicate("subject", 2, [Sorts.event, Sorts.individual])
    verb = Predicate("verb", 2, [Sorts.event, Sorts.verb])
    object_ = Predicate("object", 2, [Sorts.event, Sorts.individual])
    adjective = Predicate("adjective", 2, [Sorts.adjective, Sorts.individual])
    adverb = Predicate("adverb", 2, [Sorts.event, Sorts.adverb])


class Axioms(AxiomsBase):
    @staticmethod
    @AxiomUtils.only_one_kind_per_event(Predicates.subject)
    def axiom_only_one_subject(tableau: Tableau) -> Optional[Tableau]:
        """Only only subject per event"""

    @staticmethod
    @AxiomUtils.only_one_kind_per_event(Predicates.object_)
    def axiom_only_one_object(tableau: Tableau) -> Optional[Tableau]:
        """Only only object per event"""

    @staticmethod
    @AxiomUtils.only_one_kind_per_event(Predicates.verb)
    def axiom_only_one_verb(tableau: Tableau) -> Optional[Tableau]:
        """Only only verb per event"""