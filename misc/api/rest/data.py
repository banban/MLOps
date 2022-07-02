from typing import TypedDict


class Model(TypedDict):
    id: int
    title: str
    content: str
    category_id: int
    version: str


class ModelPayload(TypedDict, total=False):
    title: str
    content: str


class MLType(TypedDict):
    id: int
    name: str


MODELS: dict[int, Model] = {
    1: {"id": 1, "title": "Model 1.3", "category_id": 1, "content": "Content 1", "version": "3"},
    2: {"id": 2, "title": "Model 2.3", "category_id": 2, "content": "Content 2", "version": "3"},
    3: {"id": 3, "title": "Model 1.4", "category_id": 3, "content": "Content 3", "version": "4"},
    4: {"id": 4, "title": "Model 2.4", "category_id": 3, "content": "Content 4", "version": "4"},
}

MLTYPES: dict[int, MLType] = {
    1: {"id": 1, "name": "Deep Learning"},
    2: {"id": 2, "name": "XGBoost"},
    3: {"id": 3, "name": "Tensor Flow"},
}


class NotFoundError(Exception):
    pass


def all_models() -> list[Model]:
    return list(MODELS.values())


def get_model(model_id: int) -> Model:
    if not MODELS.get(model_id):
        raise NotFoundError("Model not found")
    return MODELS[model_id]


def update_model(model_id: int, payload: ModelPayload) -> Model:
    model = MODELS.get(model_id)
    if not model:
        raise NotFoundError("Model not found")
    for key, value in payload.items():
        model[key] = value
    return model


def all_categories() -> list[MLType]:
    return list(MLTYPES.values())


def get_category(category_id: int) -> MLType:
    if not MLTYPES.get(category_id):
        raise NotFoundError("MLType not found")
    return MLTYPES[category_id]
