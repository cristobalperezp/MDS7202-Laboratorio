"""
This is a boilerplate pipeline 'data_prep'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node

from lab_11.pipelines.data_prep.nodes import (
    create_model_input_table,
    get_data,
    preprocess_companies,
    preprocess_shuttles,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        nodes=[
            node(
                func=get_data,
                inputs=None,
                outputs=["companies", "shuttles", "reviews"],
                name="nodo_get_data",
            ),
            node(
                func=preprocess_companies,
                inputs="companies",
                outputs="preprocessed_companies",
                name="nodo_preprocess_companies",
            ),
            node(
                func=preprocess_shuttles,
                inputs="shuttles",
                outputs="preprocessed_shuttles",
                name="nodo_preprocess_shuttles",
            ),
            node(
                func=create_model_input_table,
                inputs=["preprocessed_companies", "preprocessed_shuttles", "reviews"],
                outputs="model_input_table",
                name="nodo_create_model_input_table",
            ),
        ]
    )
