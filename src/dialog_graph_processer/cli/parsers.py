import pathlib


def add_dgb_arguments(parser, required):
    sub_parser = parser.add_argument_group(
        "dialog_graph_build",
        "This is a group of arguments that are related to Dialog Graph Building parameters.",
    )
    sub_parser.add_argument(
        "--in-dataset",
        help="Path to the dialog dataset for loading.",
        type=pathlib.Path,
        default=pathlib.Path("data/dataset"),
        required=required,
    )
    sub_parser.add_argument(
        "--n-speakers",
        help="Number of speakers with different behaviors.",
        type=int,
        default=2,
        required=False,
    )
    sub_parser.add_argument(
        "--language",
        help="Language of data.",
        type=str,
        default="ru",
        required=False,
    )
    sub_parser.add_argument(
        "--n_cluster1",
        help="Number of clusters for first stage.",
        type=int,
        default=200,
        required=False,
    )
    sub_parser.add_argument(
        "--n_cluster2",
        help="Number of clusters for second stage.",
        type=int,
        default=30,
        required=False,
    )
    sub_parser.add_argument(
        "--out-graph",
        help="Path to the binary file with persona vector representation for storing.",
        type=pathlib.Path,
        default=None,
        required=required,
    )


def add_cand_arguments(parser, required):
    sub_parser = parser.add_argument_group(
        "candidate_scoring",
        "This is a group of arguments that are related to Dialog Graph Candidates Scoring parameters.",
    )
    sub_parser.add_argument(
        "--in-candidates",
        help="Path to the json file with candidates for loading.",
        type=pathlib.Path,
        default=pathlib.Path("data/candidates.json"),
        required=required,
    )
    sub_parser.add_argument(
        "--out-candidates",
        help="Path to the json file with candidates for storing.",
        type=pathlib.Path,
        default=None,
        required=required,
    )


def add_persona_arguments(parser, required):
    sub_parser = parser.add_argument_group(
        "persona_vectorizer",
        "This is a group of arguments that are related to Dialog Graph Persona Vectorizer Scoring parameters.",
    )
    sub_parser.add_argument(
        "--out-persona-embeddings",
        help="Path to the binary file with persona vector representation for storing.",
        type=pathlib.Path,
        default=pathlib.Path("data/persona.json"),
        required=required,
    )


def add_share_arguments(parser, required):
    sub_parser = parser.add_argument_group(
        "cand_and_persona",
        "This is a group of arguments that are related to shared parameters of Dialog Graph Candidates Scoring"
        "and Dialog Graph Persona Vectorizer Scoring.",
    )
    sub_parser.add_argument(
        "--in-graph",
        help="Path to the dialog graph directory for binary files loading.",
        type=pathlib.Path,
        default=pathlib.Path("data/graph"),
        required=required,
    )
    sub_parser.add_argument(
        "--in-dialog",
        help="Path to the json file with a dialog part for loading.",
        type=pathlib.Path,
        default=None,
        required=required,
    )
