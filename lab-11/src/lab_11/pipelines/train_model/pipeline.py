"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.18.10
"""

from kedro.pipeline import Pipeline, node

from lab_11.pipelines.train_model.nodes import evaluate_model, split_data, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Paso 1: Dividir los datos
            node(
                func=split_data,
                inputs=["model_input_table", "params:split_params"],
                outputs=[
                    "X_train",
                    "X_valid",
                    "X_test",
                    "y_train",
                    "y_valid",
                    "y_test",
                ],
                name="nodo_split_data",
            ),
            # Paso 2: Entrenar los modelos y obtener el mejor modelo
            node(
                func=train_model,
                inputs=[
                    "X_train",
                    "y_train",
                    "X_valid",
                    "y_valid",
                ],
                outputs="best_model",
                name="nodo_train_model",
            ),
            # Paso 3: Evaluar el modelo en el conjunto de test
            node(
                func=evaluate_model,
                inputs=[
                    "best_model",
                    "X_test",
                    "y_test",
                ],
                outputs=None,
                name="nodo_evaluate_model",
            ),
        ]
    )
