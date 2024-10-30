from copy import deepcopy
import copy
from dataclasses import dataclass
import functools
import operator as op
from typing import Callable, Dict, Optional, List, Tuple

from src.logic.base.syntax import Constant, Exists, Forall, Formula, Term, Variable
from src.logic.base.tableau import Tableau
from src.final_environment.logic import Predicates, Sorts


FLambda = Callable[[Term], Formula]


@dataclass
class Noun:
    name: str
    referent: Optional[str]
    is_reference: bool

    def __post_init__(self) -> None:
        if self.name:
            self.name = self.name.lower()
        if self.referent:
            self.referent = self.referent.lower()

    def __str__(self) -> str:
        return f"{self.name} ({self.referent})"


@dataclass
class Sentence:
    """This class is used for sentence initial annotation to help with translation to logic."""

    sentence: str
    subject: Noun
    verb: str
    object_: Optional[Noun]
    adjectives: List[Tuple[str, Noun]]
    is_negated: bool
    is_always: bool
    adverb: Optional[str] = None

    def __post_init__(self) -> None:
        self.sentence = self.sentence.lower()
        self.verb = self.verb.lower()
        self.adjectives = [(first.lower(), second) for first, second in self.adjectives]
        if self.adverb:
            self.adverb = self.adverb.lower()

    @classmethod
    def from_dict(cls, sentence_dict: Dict[str, any]) -> "Sentence":
        """Create and return a new sentence from a dictionary"""
        dict_ = deepcopy(sentence_dict)
        dict_["subject"] = Noun(**dict_["subject"])
        # Create proper noun objects
        if dict_.get("object_"):
            dict_["object_"] = Noun(**dict_["object_"])
        # Make sure adjectives have the correct tuple type
        adjectives = []
        for adjective, noun in dict_["adjectives"]:
            adjectives.append((adjective, Noun(**noun)))
        dict_["adjectives"] = adjectives
        # Create and return the new sentence object
        return cls(**dict_)

    def get_tableaus(self, parent: Optional[Tableau] = None) -> List[Tableau]:
        """Create all possible logical formulas from the sentence based on different readings."""
        entities = []

        # Find verb
        v_const = Sorts.verb.make_constant(self.verb)
        entities.append(v_const)
        verb = lambda e: Predicates.verb(e, v_const)

        # Find subject
        if self.subject.is_reference:
            s = Variable(Sorts.individual, self.subject.name, append_id=True)
            subject = lambda e: Exists(lambda v: Predicates.subject(e, v), variable=s)
        else:
            s_const = Sorts.individual.make_constant(self.subject.name)
            entities.append(s_const)
            if self.subject.name.startswith("a "):
                def_const = Sorts.individual.make_constant(
                    f"the {self.subject.name.split(' ')[1]}"
                )
                entities.append(def_const)
            subject = lambda e: Predicates.subject(e, s_const)

        # Find object
        object_ = None
        if self.object_:
            if self.object_.is_reference:
                o = Variable(Sorts.individual, append_id=True)
                object_ = lambda e: Exists(
                    lambda v: Predicates.object_(e, v), variable=o
                )
            else:
                o_const = Sorts.individual.make_constant(self.object_.name)
                entities.append(o_const)
                if self.object_.name.startswith("a "):
                    def_const = Sorts.individual.make_constant(
                        f"the {self.object_.name.split(' ')[1]}"
                    )
                    entities.append(def_const)
                object_ = lambda e: Predicates.object_(e, o_const)

        # Find adjectives
        adj_forms: List[Formula] = []
        for adjective, noun in self.adjectives:
            adj_const = Sorts.adjective.make_constant(adjective)
            entities.append(adj_const)
            if not noun.is_reference:
                noun_const = Sorts.individual.make_constant(noun)
                entities.append(noun_const)
                adj_forms.append(Predicates.adjective(adj_const, noun_const))
            else:
                adj_forms.append(
                    Exists(
                        lambda i: Predicates.adjective(
                            Sorts.adjective.make_constant(adjective), i
                        ),
                        variable=Variable(
                            Sorts.individual, name=noun.name, append_id=True
                        ),
                    )
                )
        adj_form = None
        if adj_forms:
            adj_form = functools.reduce(op.and_, adj_forms)

        # Collect adverbs
        adverb = None
        if self.adverb:
            adverb_const = Sorts.adverb.make_constant(self.adverb)
            adverb = lambda e: Predicates.adverb(e, adverb_const)

        # Collect formulas
        formulas = []
        # Handle the case where the sentence is negated
        if self.is_negated:
            formulas.extend(
                self._get_negated_readings(subject, verb, object_, adj_form)
            )
        else:
            # Handle the case where the sentence is not negated and has the modifier always
            if self.is_always:
                formulas.extend(
                    self._get_always_readings(subject, verb, object_, adj_form)
                )
            # Handle the case where the sentence is not negated and doesn't have the modifier always
            else:
                # In this case, there is only one simple reading of Ee. subject(e) & object(e) & adjectives
                formulas.extend(
                    self._get_simple_readings(subject, verb, object_, adverb, adj_form)
                )

        # Create a tableau for each formula / reading
        tableaus: List[Tableau] = [Tableau([formula], entities) for formula in formulas]
        # If a parent is provided, use the merge function to assert the uniqueness properties
        if parent:
            tableaus = [tableau.merge(parent=parent) for tableau in tableaus]
        # Return the created tableaus
        return tableaus

    def _get_negated_readings(
        self, subject, verb, object_, adjectives: Optional[Formula]
    ) -> List[Formula]:
        formulas = []
        if self.is_always:
            raise NotImplementedError(
                "Cannot process sentences with both always and negation."
            )
        # Technically, the powerset of the components of the sentence should be used,
        # but this is sufficient for the time being.
        # First case: negated subject formula
        formulas.append(
            Exists(lambda e: ~subject(e) & verb(e) & object_(e) & adjectives)
        )
        # Second case: negated object formula
        if object_:
            formulas.append(
                Exists(lambda e: ~object_(e) & subject(e) & verb(e) & adjectives)
            )
            # Third case: negated verg
            formulas.append(
                Exists(lambda e: ~verb(e) & subject(e) & object_(e) & adjectives)
            )
        # Fourth case: negated event
        formulas.append(
            ~Exists(lambda e: subject(e) & verb(e) & object_(e) & adjectives)
        )
        return formulas

    @staticmethod
    def _get_always_readings(
        subject, verb, object_, adjectives: Optional[Formula]
    ) -> List[Formula]:
        formulas: List[Formula] = []
        if not object_:
            raise ValueError("No object found in an always statement.")
        # First formula is subject -> (object & adjectives & verb)
        first_formula: Callable[[Term], Formula] = lambda e: object_(e) & verb(e)
        if adjectives:
            first_formula1 = lambda e: first_formula(e) & adjectives
        else:
            first_formula1 = first_formula
        formulas.append(Forall(lambda e: subject(e) >> first_formula1(e)))
        # Second formula is object -> (subject & adjectives & verb)
        second_formula: FLambda = lambda e: subject(e) & verb(e)
        if adjectives:
            second_formula1 = lambda e: second_formula(e) & adjectives
        else:
            second_formula1 = second_formula
        formulas.append(Forall(lambda e: object_(e) >> second_formula1(e)))
        # Third formula is verb -> (subject & adjectives & object)
        third_formula: FLambda = lambda e: subject(e) & object_(e)
        if adjectives:
            third_formula1 = lambda e: third_formula(e) & adjectives
        else:
            third_formula1 = third_formula

        formulas.append(Forall(lambda e: verb(e) >> third_formula1(e)))
        return formulas

    @staticmethod
    def _get_simple_readings(
        subject: FLambda,
        verb: FLambda,
        object_: Optional[FLambda],
        adverb: Optional[FLambda],
        adjectives: Optional[Formula],
    ) -> List[Formula]:
        formula: FLambda = lambda e: verb(e) & subject(e)
        if object_:
            formula1 = lambda e: formula(e) & object_(e)
        else:
            formula1 = formula

        if adjectives:
            formula2 = lambda e: formula1(e) & adjectives
        else:
            formula2 = formula1

        if adverb:
            formula3 = lambda e: formula2(e) & adverb(e)
        else:
            formula3 = formula2

        return [Exists(formula3, Sorts.event)]


def term_to_word(term: Constant, rename_variables: bool = False) -> str:
    if not term.name.startswith("_"):
        return term.name
    if rename_variables:
        return f"some {term.sort.name}"
    return f"{term.sort.name}{term.name}"


def tableau_to_sentence_str(tableau: Tableau, rename_variables: bool = False) -> str:
    """Create a sentence from a tableau model of a sentence"""
    subject = ""
    verb = ""
    object_ = ""
    adj = ""
    adverb = ""

    for literal in tableau.branch_literals:
        match literal.predicate:
            case Predicates.subject:
                subject = term_to_word(literal.args[1], rename_variables)
            case Predicates.verb:
                verb = term_to_word(literal.args[1], rename_variables)
            case Predicates.object_:
                object_ = term_to_word(literal.args[1], rename_variables)
            case Predicates.adjective:
                subject = term_to_word(literal.args[1], rename_variables)
                adj = term_to_word(literal.args[0], rename_variables)
            case Predicates.adverb:
                adverb = term_to_word(literal.args[1], rename_variables)
            case _:
                raise NotImplementedError()
    return " ".join(f"{subject} {verb} {adj} {object_} {adverb}".split())


def tableau_to_sentence(tableau: Tableau) -> Sentence:
    """Create a sentence from a tableau model of a sentence"""
    subject = None
    verb = None
    object_ = None
    adjectives = []
    adverb = None
    for literal in tableau.branch_literals:
        match literal.predicate:
            case Predicates.subject:
                subject = Noun(term_to_word(literal.args[1]), None, False)
            case Predicates.verb:
                verb = term_to_word(literal.args[1])
            case Predicates.object_:
                object_ = Noun(term_to_word(literal.args[1]), None, False)
            case Predicates.adjective:
                noun = Noun(term_to_word(literal.args[1]), None, False)
                adj = term_to_word(literal.args[0])
                adjectives.append((adj, noun))
            case Predicates.adverb:
                adverb = term_to_word(literal.args[1])
    sentence_str = tableau_to_sentence_str(tableau)
    return Sentence(
        sentence_str, subject, verb, object_, adjectives, False, False, adverb=adverb
    )
