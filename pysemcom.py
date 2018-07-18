# -*- coding: utf-8 -*-
"""Common Entry Point for the different subprograms of pyComplex"""
import sys
from abc import ABCMeta
from collections import namedtuple

from subprograms.loader import SubProgramLoader

SubProgram = namedtuple('SubProgram', ['name', 'subprogram', 'description'])


class PyComplex(metaclass=ABCMeta):

    @classmethod
    def _get_description(cls):
        return """pysemcom (pySemanticComplexity) allow to compute from a bunch of different text files lexical,
        syntactical and semantic complexity as vectors for each of these texts."""

    @classmethod
    def _get_epilog(cls, progname):
        epilog = """Semantic complexity relies on DBpedia Entity Recognition Graph computation based on multiple 
ontologies used in DBPedia. PyComplex offers several subprograms to process multiples files in parallels.
The Subprograms are :\n"""
        for sp in SubProgramLoader().available_subprograms():
            epilog += ("\t- %s: %s\n" % (sp.name, sp.description))
        epilog += ("Subprogram respective help can be obtained by executing\n %s <SUB_PROGRAM> --help" % progname)
        return epilog

    @classmethod
    def _print_help(cls, progname):
        print('usage: %s [-h] subprogram' % progname)

    @classmethod
    def _print_full_help(cls, progname):
        cls._print_help(progname)
        print()
        print(cls._get_description())
        print()
        print("optionnal arguments:")
        print("  -h, --help show this help message and exit")
        print()
        print(cls._get_epilog(progname))

    @classmethod
    def _print_bad_choice(cls, progname, bad_choice, good_choices):
        str_choices = ", ".join(good_choices)
        cls._print_help(progname)
        print("%s: error argument subprogram: invalid choice: '%s' (choose from %s)" % (progname,
                                                                                        bad_choice, str_choices))

    @classmethod
    def start(cls):
        progname_base = sys.argv[0]

        if len(sys.argv) < 2:
            cls._print_help(progname_base)
            sys.exit(1)

        if len(sys.argv) == 2 and sys.argv[1] in ['-h', '--help']:
            cls._print_full_help(progname_base)
            sys.exit(0)

        subprog_name = list(filter(lambda x: not x.startswith('-'), sys.argv[1:]))
        subprog_name = subprog_name[0] if subprog_name and subprog_name[
            0] in SubProgramLoader().available_names() else None

        if subprog_name is None:
            cls._print_bad_choice(progname_base, sys.argv[1], SubProgramLoader().available_names())
            sys.exit(1)

        # remove the positionnal argument
        sys.argv = [arg for arg in sys.argv if arg != subprog_name]

        # sub prog name to give for printing
        subprog_name_printout = "%s %s" % (progname_base, subprog_name)

        # load sub prg
        sub_prg = SubProgramLoader().load_subprogram_class(subprog_name)(progname=subprog_name_printout)
        # start sub prg
        sub_prg.start()


if __name__ == '__main__':
    PyComplex.start()
