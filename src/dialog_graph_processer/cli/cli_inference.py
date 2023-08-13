# %%
# %load_ext autoreload
# %autoreload 2
# %%

import argparse
import pathlib
import sys
import json

from .parsers import add_dgb_arguments, add_cand_arguments, add_persona_arguments, add_share_arguments

from dialog_graph_processer.DGAC_MP.intent_prediction import IntentPredictor
from dialog_graph_processer.Ranking.ranking import Ranking


def get_parser():
    """
    This function parse_args is designed to parse the command line arguments passed to a Python script.
    It uses the argparse module to define the command line interface.
    """
    parser = argparse.ArgumentParser(description="Dialog Graph Inference CLI API.")
    modes = {
        "dgb": {"arg": "--out-graph"},
        "cand": {"arg": "--out-candidates"},
        "persona": {"arg": "--out-persona-embeddings"},
    }

    for mode_props in modes.values():
        mode_props["used"] = mode_props["arg"] in sys.argv

    add_dgb_arguments(parser, modes["dgb"]["used"])
    add_cand_arguments(parser, modes["cand"]["used"])
    add_persona_arguments(parser, modes["persona"]["used"])
    add_share_arguments(parser, modes["cand"]["used"] | modes["persona"]["used"])

    if sum([mode_props["used"] for mode_props in modes.values()], 0) != 1:
        parser.print_help(sys.stderr)
        print(f"Can be used only one from {[mode_props['arg'] for mode_props in modes.values()]}")
        sys.exit(1)

    return parser


def run_dg_build(
    dataset_dir: pathlib.Path,
    save_dir: pathlib.Path,
    n_cluster1: int = 200,
    n_cluster2: int = 30,
    language: str = "ru",
    n_speakers: int = 2,
):
    save_dir.mkdir(exist_ok=True)
    embeddings_file = save_dir / "ru_embeddings.npy"
    dialog_graph_file = save_dir / "dialog_graph.pickle"

    intent_predictor = IntentPredictor(
        str(list(dataset_dir.glob("*.json"))[0]),
        str(embeddings_file),
        language,
        n_speakers,
        [n_cluster1, n_cluster2],
    )

    intent_predictor.dialog_graph_auto_construction()
    intent_predictor.dump_dialog_graph(str(dialog_graph_file))
    intent_predictor.dgl_graphs_preprocessing()
    intent_predictor.init_message_passing_model(str(save_dir))


def run_candidates_scoring(
    graph_dir: pathlib.Path,
    dialog_file: pathlib.Path,
    candidates_file: pathlib.Path,
    out_candidates_file: pathlib.Path,
):
    out_candidates_file.parent.mkdir(exist_ok=True)
    GRAPH_PATH = str(graph_dir / "dialog_graph.pickle")
    USER_MP_PATH = str(graph_dir / "GAT_system")
    SYSTEM_MP_PATH = str(graph_dir / "GAT_user")
    MODEL_PATHS = [USER_MP_PATH, SYSTEM_MP_PATH]

    ranking = Ranking(GRAPH_PATH, MODEL_PATHS)

    dialog = json.load(dialog_file.open("rt"))

    candidates = json.load(candidates_file.open("rt"))

    ranked_candidates = ranking.ranking(dialog, candidates)

    json.dump(
        [utterance for utterance, _ in ranked_candidates],
        out_candidates_file.open("rt"),
        ensure_ascii=False,
        indent=2,
    )


def cli():
    args = None
    parser = get_parser()
    args = parser.parse_args(args)
    if args.out_graph:
        run_dg_build(
            args.in_dataset,
            args.out_graph,
            n_cluster1=args.n_cluster1,
            n_cluster2=args.n_cluster2,
            language=args.language,
            n_speakers=args.n_speakers,
        )
    elif args.out_candidates:
        run_candidates_scoring(
            args.in_graph,
            args.in_dialog,
            args.in_candidates,
            args.in_graph,
            args.out_candidates,
        )
    elif args.out_persona_embeddings:
        pass
